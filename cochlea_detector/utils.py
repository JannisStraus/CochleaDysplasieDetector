import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import SimpleITK as sitk
from cv2.typing import MatLike
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

BBox = tuple[int, int, int, int] | None
logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    cache_dir = Path.home() / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def save_bbox(
    image_path: str | Path,
    output_path: str | Path,
    bbox: BBox,
    cls: str,
) -> None:
    img = draw_bbox(
        image_path,
        bbox,
        cls,
    )
    cv2.imwrite(str(output_path), img)


def draw_bbox(
    image_path: str | Path,
    bbox: BBox,
    cls: str,
) -> MatLike:
    img = cv2.imread(str(image_path))
    if bbox is None:
        return img
    p1, p2 = bbox[:2], bbox[2:]
    img = cv2.rectangle(img, p1, p2, color=(0, 255, 0), thickness=4)
    img = cv2.putText(
        img,
        cls,
        (p1[0], max(0, p1[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        thickness=4,
    )
    return img


def fenstern(
    arr: NDArray[np.float32], low: float = -1024.0, high: float = 3071.0
) -> NDArray[np.uint8]:
    array = np.clip((arr - low) / (high - low), 0.0, 1.0)
    array = np.rint(array * 255.0).astype(np.uint8)
    return array


def nifti_to_png(src: Path, dst: Path, overwrite: bool = True) -> list[Path]:
    if dst and not overwrite:
        out_paths = sorted(dst.glob("nifti_*.png"))
        if out_paths:
            return out_paths
    img = sitk.ReadImage(src)
    arr = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)
    arr = fenstern(arr, -200.0, 400.0)
    out_paths = []
    for z in range(arr.shape[0]):
        png = Image.fromarray(arr[z, :, :], mode="L")
        out_path = dst / f"nifti_{z:03d}.png"
        png.save(out_path)
        out_paths.append(out_path)
    return out_paths


def tensor_to_list(t: Tensor, dtype: type | None = None) -> list[Any]:
    arr = t.detach().cpu().numpy()
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr.tolist()  # type: ignore
