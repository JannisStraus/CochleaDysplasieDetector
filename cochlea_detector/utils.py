import logging
from pathlib import Path

import cv2
from cv2.typing import MatLike

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
