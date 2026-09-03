import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from tqdm import tqdm

from cochlea_detector.download import download_weights, resolve_release
from cochlea_detector.utils import (
    CLASSES,
    SIDES,
    CaseResult,
    Side,
    collect_cases,
    side_of,
    write_results,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_model(
    weights_dir: Path, n_folds: int, checkpoint: str, device: str | None = None
) -> nnUNetPredictor:
    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(
        "Loading %s fold(s) from model %s on %s...",
        n_folds,
        weights_dir.parent.name,
        torch_device,
    )
    predictor = nnUNetPredictor(
        use_mirroring=False,
        device=torch_device,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(weights_dir), use_folds=tuple(range(n_folds)), checkpoint_name=checkpoint
    )
    return predictor


def evaluate(seg_file: str | Path) -> tuple[str | None, str | None]:
    """Return the largest predicted class per ear in a segmentation mask."""
    arr = sitk.GetArrayViewFromImage(sitk.ReadImage(seg_file))
    labels = arr[arr > 0]
    if labels.size == 0:
        return None, None

    unique, counts = np.unique(labels, return_counts=True)
    # Segmentation labels are 1-based indices into CLASSES.
    best: dict[Side, tuple[str | None, int]] = dict.fromkeys(SIDES, (None, 0))
    for label, count in zip(unique.tolist(), counts.tolist(), strict=True):
        if not 1 <= label <= len(CLASSES):
            logger.warning(
                "Ignoring unknown segmentation label %s in %s", label, seg_file
            )
            continue
        class_name = CLASSES[label - 1]
        side = side_of(class_name)
        if count > best[side][1]:
            best[side] = (class_name, count)

    return best["right"][0], best["left"][0]


def infer(
    input_file_or_dir: str | Path,
    output_dir: str | Path,
    device: str | None = None,
    *,
    release: str | None = None,
    weights_dir: str | Path | None = None,
) -> None:
    input_file_or_dir = Path(input_file_or_dir)
    output_dir = Path(output_dir)
    spec = resolve_release("nnunet", release)
    weights = Path(weights_dir) if weights_dir else download_weights("nnunet", release)
    model = load_model(weights, spec.n_folds, spec.checkpoint, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(input_file_or_dir)
    if not cases:
        logger.warning("No NIfTI files found in %s", input_file_or_dir)

    seg_dir = output_dir / "segmentations"
    seg_dir.mkdir(parents=True, exist_ok=True)
    if cases:
        model.predict_from_files(
            [[str(case.path)] for case in cases],
            [str(seg_dir / case.stem) for case in cases],
            save_probabilities=False,
            overwrite=True,
        )

    # Evaluation
    results = [
        CaseResult(case.stem, *evaluate(seg_dir / f"{case.stem}.nii.gz"))
        for case in tqdm(cases, desc="Evaluating")
    ]

    # CSV
    write_results(results, output_dir / "results.csv")
