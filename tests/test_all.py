import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import numpy.testing as npt

from cochlea_detector.utils import draw_bbox, save_bbox


def write_checkerboard(path: Path, height: int = 250, width: int = 750) -> None:
    tile = 50
    yy, xx = np.indices((height, width))
    checker = ((xx // tile + yy // tile) % 2).astype(np.uint8)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[checker == 0] = (255, 0, 0)
    img[checker == 1] = (0, 0, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


class TestDrawBBox(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.img_path = self.tmp_dir / "input.png"
        write_checkerboard(self.img_path, 500, 500)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_no_bbox(self) -> None:
        expected = cv2.imread(str(self.img_path))
        npt.assert_array_equal(draw_bbox(self.img_path, None, "test"), expected)

    def test_box_only(self) -> None:
        img = draw_bbox(self.img_path, (100, 100, 200, 200), "test")
        original = cv2.imread(str(self.img_path))
        self.assertEqual(img.shape, original.shape)
        green = np.array([0, 255, 0], dtype=np.uint8)
        npt.assert_array_equal(
            img[100:102, 100:201], np.broadcast_to(green, (2, 101, 3))
        )
        npt.assert_array_equal(
            img[100:201, 100:102], np.broadcast_to(green, (101, 2, 3))
        )

    def test_save_bbox(self) -> None:
        output = self.tmp_dir / "out.png"
        save_bbox(self.img_path, output, (10, 10, 20, 20), "test")
        self.assertTrue(output.is_file())
        self.assertIsNotNone(cv2.imread(str(output)))
