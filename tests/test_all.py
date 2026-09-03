import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import numpy.testing as npt
from PIL import Image

from cochlea_detector.utils import (
    CLASSES,
    Case,
    CaseResult,
    collect_cases,
    draw_bbox,
    nifti_stem,
    save_bbox,
    side_of,
    write_results,
)


def write_checkerboard(path: Path, height: int = 250, width: int = 750) -> None:
    tile = 50
    yy, xx = np.indices((height, width))
    checker = ((xx // tile + yy // tile) % 2).astype(np.uint8)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[checker == 0] = (255, 0, 0)
    arr[checker == 1] = (0, 0, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)
    img.save(path)


class TestDrawBBox(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.img_path = self.tmp_dir / "input.png"
        write_checkerboard(self.img_path, 500, 500)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_no_bbox(self) -> None:
        expected = Image.open(self.img_path).convert("RGB")
        npt.assert_array_equal(
            np.asarray(draw_bbox(self.img_path, None, "test")), np.asarray(expected)
        )

    def test_box_only(self) -> None:
        drawn = draw_bbox(self.img_path, (100, 100, 200, 200), "test")
        original = Image.open(self.img_path)
        self.assertEqual(drawn.size, original.size)
        # Pillow draws the outline inwards, so a width of 4 covers 100..103.
        img = np.asarray(drawn)
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
        with Image.open(output) as saved:
            self.assertEqual(saved.size, (500, 500))


class TestSideOf(unittest.TestCase):
    def test_every_class_has_a_side(self) -> None:
        for class_name in CLASSES:
            with self.subTest(class_name=class_name):
                self.assertIn(side_of(class_name), ("right", "left"))

    def test_side_follows_the_name_not_the_index(self) -> None:
        # Guards against a reordering of CLASSES silently swapping the ears.
        for class_name in CLASSES:
            expected = "right" if class_name.endswith("_re") else "left"
            self.assertEqual(side_of(class_name), expected)

    def test_unknown_suffix_raises(self) -> None:
        with self.assertRaises(ValueError):
            side_of("dysplasie")


class TestNiftiStem(unittest.TestCase):
    def test_strips_known_suffixes(self) -> None:
        self.assertEqual(nifti_stem(Path("a/patient_01.nii.gz")), "patient_01")
        self.assertEqual(nifti_stem(Path("a/patient_01.NII.GZ")), "patient_01")
        self.assertEqual(nifti_stem(Path("a/patient_01.nii")), "patient_01")

    def test_rejects_other_files(self) -> None:
        self.assertIsNone(nifti_stem(Path("a/patient_01.gz")))
        self.assertIsNone(nifti_stem(Path("a/patient_01.dcm")))


class TestCollectCases(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_filters_and_sorts(self) -> None:
        for name in ("b.nii.gz", "a.nii", "notes.txt", "c.dcm"):
            (self.tmp_dir / name).touch()
        (self.tmp_dir / "subdir.nii.gz").mkdir()
        cases = collect_cases(self.tmp_dir)
        self.assertEqual([case.stem for case in cases], ["a", "b"])

    def test_single_file(self) -> None:
        path = self.tmp_dir / "one.nii.gz"
        path.touch()
        self.assertEqual(collect_cases(path), [Case("one", path)])

    def test_directory_without_niftis(self) -> None:
        (self.tmp_dir / "notes.txt").touch()
        self.assertEqual(collect_cases(self.tmp_dir), [])


class TestWriteResults(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.csv_file = Path(self._tmp.name) / "nested" / "results.csv"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_empty_results_still_write_a_header(self) -> None:
        write_results([], self.csv_file)
        rows = list(csv.reader(self.csv_file.read_text().splitlines()))
        self.assertEqual(rows, [["image_name", "pred_right", "pred_left"]])

    def test_rows_round_trip(self) -> None:
        write_results(
            [
                CaseResult("a", "normal_re", "dysplasie_li"),
                CaseResult("b", None, None),
            ],
            self.csv_file,
        )
        rows = list(csv.DictReader(self.csv_file.read_text().splitlines()))
        self.assertEqual(rows[0]["pred_right"], "normal_re")
        self.assertEqual(rows[1]["pred_left"], "")
