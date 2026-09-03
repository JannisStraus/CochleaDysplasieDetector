import hashlib
import json
import logging
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import requests
from tqdm import tqdm

from cochlea_detector.utils import get_cache_dir

logger = logging.getLogger(__name__)


class ChecksumError(RuntimeError):
    pass


class WeightsLayoutError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelRelease:
    url: str
    sha256: str | None
    n_folds: int
    checkpoint: str


@dataclass(frozen=True)
class ModelEntry:
    default: str
    releases: dict[str, ModelRelease]


@lru_cache(maxsize=1)
def load_registry() -> dict[str, ModelEntry]:
    raw = json.loads(
        files("cochlea_detector").joinpath("models.json").read_text("utf-8")
    )
    return {
        model: ModelEntry(
            default=entry["default"],
            releases={
                name: ModelRelease(
                    url=spec["url"],
                    sha256=spec["sha256"],
                    n_folds=spec["n_folds"],
                    checkpoint=spec["checkpoint"],
                )
                for name, spec in entry["releases"].items()
            },
        )
        for model, entry in raw.items()
    }


def available_models() -> tuple[str, ...]:
    return tuple(sorted(load_registry()))


def resolve_model(model: str = "yolo") -> ModelEntry:
    registry = load_registry()
    try:
        return registry[model]
    except KeyError:
        known = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown model '{model}'. Available models: {known}."
        ) from None


def resolve_release(model: str = "yolo", release: str | None = None) -> ModelRelease:
    entry = resolve_model(model)
    name = release or entry.default
    try:
        return entry.releases[name]
    except KeyError:
        known = ", ".join(sorted(entry.releases))
        raise ValueError(
            f"Unknown weights release '{name}' for model '{model}'. "
            f"Available releases: {known}."
        ) from None


def weights_exist(weights_dir: str | Path, n_folds: int, checkpoint: str) -> bool:
    weights_dir = Path(weights_dir)
    return n_folds > 0 and all(
        (weights_dir / f"fold_{i}" / checkpoint).is_file() for i in range(n_folds)
    )


def unzip(zip_file: str | Path, output_dir: str | Path) -> None:
    zip_file = Path(zip_file)
    with zipfile.ZipFile(zip_file, "r", metadata_encoding="utf-8") as zip_ref:
        zip_ref.extractall(output_dir)


def download(
    dst_file: str | Path,
    url: str,
    params: dict[str, str] | None = None,
    *,
    sha256: str | None = None,
) -> None:
    dst_file = Path(dst_file)
    if sha256 is None:
        logger.warning("No checksum published for '%s'.", url)
    response = requests.get(url, params=params, stream=True, timeout=20.0)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    digest = hashlib.sha256()

    with (
        dst_file.open("wb") as file,
        tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Download"
        ) as progress_bar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                file.write(chunk)
                digest.update(chunk)
                progress_bar.update(len(chunk))

    if sha256 is not None and digest.hexdigest() != sha256.lower():
        raise ChecksumError(
            f"Checksum mismatch for '{url}': expected sha256 {sha256.lower()}, "
            f"got {digest.hexdigest()}. The archive was corrupted in transit or "
            f"replaced at the source; refusing to extract it."
        )


def download_and_unzip(
    dst_dir: Path,
    url: str,
    params: dict[str, str] | None = None,
    *,
    sha256: str | None = None,
) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    zip_file = dst_dir.with_suffix(".zip")
    try:
        download(zip_file, url, params, sha256=sha256)
        unzip(zip_file, dst_dir.parent)
    finally:
        zip_file.unlink(missing_ok=True)
    return dst_dir


def download_weights(model: str = "yolo", release: str | None = None) -> Path:
    entry = resolve_model(model)
    name = release or entry.default
    spec = resolve_release(model, name)
    weights_dir = (
        get_cache_dir() / "CochleaDysplasieDetector" / model / name / "weights"
    )
    if weights_exist(weights_dir, spec.n_folds, spec.checkpoint):
        return weights_dir

    logger.info("Downloading %s weights release '%s'.", model, name)
    download_and_unzip(weights_dir, spec.url, sha256=spec.sha256)
    if not weights_exist(weights_dir, spec.n_folds, spec.checkpoint):
        raise WeightsLayoutError(
            f"The archive for {model} release '{name}' did not provide "
            f"fold_0..fold_{spec.n_folds - 1}/{spec.checkpoint} under "
            f"'{weights_dir}'. The release layout does not match models.json."
        )
    return weights_dir
