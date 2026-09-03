import csv
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypeVar, overload

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

Side = Literal["right", "left"]
SIDES: tuple[Side, ...] = ("right", "left")

CLASSES = [
    "dysplasie_re",
    "dysplasie_li",
    "normal_re",
    "normal_li",
]
# The ear a class belongs to is read off its name suffix, never off the parity
# of its index: reordering CLASSES must not silently swap left and right.
SIDE_SUFFIXES: dict[str, Side] = {"_re": "right", "_li": "left"}
NIFTI_SUFFIXES = (".nii.gz", ".nii")  # longest first
# HU window used when exporting slices to PNG. This has to match the window the
# detector was trained on - changing it changes the contrast the model sees.
PNG_WINDOW = (-200.0, 400.0)

BBox = tuple[int, int, int, int] | None
T = TypeVar("T")
logger = logging.getLogger(__name__)


class Case(NamedTuple):
    """One input volume: its name without the NIfTI suffix, and its path."""

    stem: str
    path: Path


@dataclass(frozen=True)
class CaseResult:
    """Per-volume prediction, one row of ``results.csv``."""

    image_name: str
    pred_right: str | None
    pred_left: str | None


def side_of(class_name: str) -> Side:
    """Return the ear ``class_name`` describes, based on its name suffix."""
    for suffix, side in SIDE_SUFFIXES.items():
        if class_name.endswith(suffix):
            return side
    known = ", ".join(sorted(SIDE_SUFFIXES))
    raise ValueError(
        f"Class '{class_name}' carries no side suffix (expected one of: {known})."
    )


def get_cache_dir() -> Path:
    """Return the per-user cache root for the current platform."""
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data)
        return Path.home() / "AppData" / "Local"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home)
    return Path.home() / ".cache"


def nifti_stem(path: Path) -> str | None:
    """Return the filename without its NIfTI suffix, or None if it isn't one."""
    lower = path.name.lower()
    for suffix in NIFTI_SUFFIXES:
        if lower.endswith(suffix):
            return path.name[: -len(suffix)]
    return None


def collect_cases(input_file_or_dir: Path) -> list[Case]:
    """Return the NIfTI volumes under ``input_file_or_dir``, sorted by name."""
    if input_file_or_dir.is_file():
        candidates = [input_file_or_dir]
    else:
        candidates = sorted(input_file_or_dir.iterdir())

    cases = []
    for path in candidates:
        stem = nifti_stem(path)
        if stem is None or not path.is_file():
            logger.info("Skipping %s", path.name)
            continue
        cases.append(Case(stem, path))
    return cases


def write_results(results: Sequence[CaseResult], csv_file: Path) -> None:
    """Write ``results`` to ``csv_file``, always emitting a header row."""
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[f.name for f in fields(CaseResult)],
            lineterminator="\n",  # match the LF output pandas produced
        )
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def save_bbox(
    image_path: str | Path,
    output_path: str | Path,
    bbox: BBox,
    cls: str,
) -> None:
    draw_bbox(image_path, bbox, cls).save(output_path)


def draw_bbox(image_path: str | Path, bbox: BBox, cls: str) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    if bbox is None:
        return img
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline=(0, 255, 0), width=4)
    draw.text(
        (bbox[0], max(0, bbox[1] - 40)),
        cls,
        fill=(0, 255, 0),
        font=ImageFont.load_default(size=32),
    )
    return img


def apply_window(
    arr: NDArray[np.float32], low: float = -1024.0, high: float = 3071.0
) -> NDArray[np.uint8]:
    """Map Hounsfield units in ``[low, high]`` onto the 0-255 grey range."""
    array = np.clip((arr - low) / (high - low), 0.0, 1.0)
    windowed: NDArray[np.uint8] = np.rint(array * 255.0).astype(np.uint8)
    return windowed


def nifti_to_png(src: Path, dst: Path, overwrite: bool = True) -> list[Path]:
    """Export every axial slice of ``src`` as a PNG in ``dst``."""
    if not overwrite:
        existing = sorted(dst.glob("nifti_*.png"))
        if existing:
            return existing
    img = sitk.ReadImage(src)
    arr = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)
    windowed = apply_window(arr, *PNG_WINDOW)
    dst.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for z in range(windowed.shape[0]):
        png = Image.fromarray(windowed[z, :, :], mode="L")
        out_path = dst / f"nifti_{z:03d}.png"
        png.save(out_path)
        out_paths.append(out_path)
    return out_paths


@overload
def tensor_to_list(t: Tensor, dtype: type[T]) -> list[T]: ...
@overload
def tensor_to_list(t: Tensor, dtype: None = None) -> list[Any]: ...
def tensor_to_list(t: Tensor, dtype: type[Any] | None = None) -> list[Any]:
    arr = t.detach().cpu().numpy()
    if dtype is not None:
        arr = arr.astype(dtype)
    values: list[Any] = arr.tolist()
    return values
