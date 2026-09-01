import contextlib
import logging
import tempfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from cochlea_detector.download import download_weights, resolve_release
from cochlea_detector.utils import nifti_to_png, tensor_to_list

ALLOWED_FORMAT = {
    ".nii",
    ".gz",
}
CLASSES = [
    "dysplasie_re",
    "dysplasie_li",
    "normal_re",
    "normal_li",
]
MAP_CLASSES = dict(enumerate(CLASSES))

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_model(weights_dir: Path, n_folds: int = 5) -> tuple[YOLO, ...]:
    logger.info("Loading %s folds from model %s...", n_folds, weights_dir)
    return tuple(
        YOLO(weights_dir / f"fold_{i}" / "best.pt", verbose=True)
        for i in range(n_folds)
    )


def measure_results(
    models: tuple[YOLO, ...], pngs: list[Path], device: str | None = None
) -> tuple[str | None, str | None]:
    # TODO: add multiple fold logic
    n_folds = len(models)
    if len(models) > 1:
        raise NotImplementedError(f"Models with {n_folds} folds are not supported yet.")
    model = models[0]
    results = model(pngs, conf=0.2, verbose=False, device=device)
    right_classes = []
    left_classes = []
    for result in results:
        boxes = result.boxes
        if boxes is None or boxes.conf.numel() == 0:
            continue

        confs_list = tensor_to_list(boxes.conf, float)
        classes_list = tensor_to_list(boxes.cls, int)

        # build detection dicts per slice
        detections = [
            {"cls": cls, "conf": conf, "class_name": MAP_CLASSES[cls]}
            for cls, conf in zip(classes_list, confs_list, strict=False)
            if cls in MAP_CLASSES
        ]

        # select best right (even) and best left (odd)
        right_dets = [d for d in detections if d["cls"] % 2 == 0]
        left_dets = [d for d in detections if d["cls"] % 2 == 1]

        best_right = max(right_dets, key=lambda d: d["conf"]) if right_dets else None
        best_left = max(left_dets, key=lambda d: d["conf"]) if left_dets else None

        if best_right:
            right_classes.append(best_right["class_name"])
        if best_left:
            left_classes.append(best_left["class_name"])

    # determine most frequent class per side
    most_common_right = Counter(right_classes).most_common(1)
    most_common_left = Counter(left_classes).most_common(1)
    pred_right = most_common_right[0][0] if most_common_right else None
    pred_left = most_common_left[0][0] if most_common_left else None

    return pred_right, pred_left


def infer(
    input_file_or_dir: str | Path,
    output_dir: str | Path,
    device: str | None = None,
    save_pngs: bool = True,
) -> None:
    input_file_or_dir = Path(input_file_or_dir)
    output_dir = Path(output_dir)
    weights_dir = download_weights()
    model = load_model(weights_dir, resolve_release().n_folds)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dict: dict[str, list[str | float | None]] = defaultdict(list)

    if input_file_or_dir.is_file():
        files = [input_file_or_dir]
    else:
        files = list(input_file_or_dir.iterdir())

    for image_path in tqdm(files, desc="Predicting"):
        name = image_path.name
        if not image_path.is_file() or image_path.suffix.lower() not in ALLOWED_FORMAT:
            logger.info("Skipping %s", name)
            continue

        # Preprocessing
        with contextlib.ExitStack() as stack:
            if save_pngs:
                png_dir = output_dir / "pngs" / image_path.name
                png_dir.mkdir(parents=True, exist_ok=True)
            else:
                png_dir = Path(
                    stack.enter_context(tempfile.TemporaryDirectory(prefix="cochlea-"))
                )
            pngs = nifti_to_png(image_path, png_dir)

            # Inference
            pred_right, pred_left = measure_results(model, pngs, device)

            result_dict["image_name"].append(name)
            result_dict["pred_right"].append(pred_right)
            result_dict["pred_left"].append(pred_left)
        break

    # CSV
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(output_dir / "results.csv", index=False)
