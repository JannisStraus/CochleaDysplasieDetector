import contextlib
import logging
import tempfile
from collections import Counter
from functools import lru_cache
from operator import itemgetter
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

from cochlea_detector.download import download_weights, resolve_release
from cochlea_detector.utils import (
    CLASSES,
    SIDES,
    CaseResult,
    Side,
    collect_cases,
    nifti_to_png,
    side_of,
    tensor_to_list,
    write_results,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_model(
    weights_dir: Path, n_folds: int, checkpoint: str = "best.pt"
) -> tuple[YOLO, ...]:
    logger.info("Loading %s fold(s) from model %s...", n_folds, weights_dir)
    return tuple(
        YOLO(weights_dir / f"fold_{i}" / checkpoint, verbose=True)
        for i in range(n_folds)
    )


def _most_common(class_names: list[str]) -> str | None:
    ranked = Counter(class_names).most_common(1)
    return ranked[0][0] if ranked else None


def evaluate(
    models: tuple[YOLO, ...],
    pngs: list[Path],
    device: str | None = None,
    conf: float = 0.2,
) -> tuple[str | None, str | None]:
    model = models[0]
    results = model(pngs, conf=conf, verbose=False, device=device)
    per_side: dict[Side, list[str]] = {side: [] for side in SIDES}
    for result in results:
        boxes = result.boxes
        if boxes is None or boxes.conf.numel() == 0:
            continue

        confs_list = tensor_to_list(boxes.conf, float)
        classes_list = tensor_to_list(boxes.cls, int)

        # build (class name, confidence) detections per slice
        detections: list[tuple[str, float]] = []
        for cls, conf_value in zip(classes_list, confs_list, strict=True):
            if not 0 <= cls < len(CLASSES):
                logger.warning("Ignoring detection with unknown class id %s", cls)
                continue
            detections.append((CLASSES[cls], conf_value))

        # keep the most confident detection per ear for this slice
        for side in SIDES:
            side_dets = [d for d in detections if side_of(d[0]) == side]
            if side_dets:
                per_side[side].append(max(side_dets, key=itemgetter(1))[0])

    # determine most frequent class per side
    return _most_common(per_side["right"]), _most_common(per_side["left"])


def infer(
    input_file_or_dir: str | Path,
    output_dir: str | Path,
    device: str | None = None,
    *,
    release: str | None = None,
    weights_dir: str | Path | None = None,
    save_pngs: bool = True,
) -> None:
    input_file_or_dir = Path(input_file_or_dir)
    output_dir = Path(output_dir)
    spec = resolve_release("yolo", release)
    # TODO: add multiple fold logic
    if spec.n_folds > 1:
        raise NotImplementedError(
            f"Models with {spec.n_folds} folds are not supported yet."
        )
    weights = Path(weights_dir) if weights_dir else download_weights("yolo", release)
    models = load_model(weights, spec.n_folds, spec.checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(input_file_or_dir)
    if not cases:
        logger.warning("No NIfTI files found in %s", input_file_or_dir)

    results: list[CaseResult] = []
    for case in tqdm(cases, desc="Predicting"):
        # Preprocessing
        with contextlib.ExitStack() as stack:
            if save_pngs:
                png_dir = output_dir / "pngs" / case.stem
            else:
                png_dir = Path(
                    stack.enter_context(tempfile.TemporaryDirectory(prefix="cochlea-"))
                )
            pngs = nifti_to_png(case.path, png_dir)

            # Inference and evaluation
            pred_right, pred_left = evaluate(models, pngs, device)
        results.append(CaseResult(case.stem, pred_right, pred_left))

    # CSV
    write_results(results, output_dir / "results.csv")
