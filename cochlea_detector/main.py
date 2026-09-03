import argparse
import logging
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from cochlea_detector.download import available_models

logger = logging.getLogger("cochlea_detector")


def _existing(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"'{path}' does not exist")
    return path


def _target(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _setup_logging(verbose: bool) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect and classify Cochlea Dysplasie in CT images."
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {version('cochlea-detector')}",
    )
    parser.add_argument(
        "model",
        nargs="?",
        choices=available_models(),
        default="yolo",
        help="Model used for inference (default: %(default)s).",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=_existing,
        required=True,
        help="Path to an input NIfTI file or a directory of them.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=_target,
        required=True,
        help="Path to save detection results.",
    )
    parser.add_argument(
        "-d",
        "--device",
        required=False,
        help="Device for inference (e.g. 'cuda' or 'cpu'). Default: auto-detect.",
    )
    # parser.add_argument(
    #     "-r",
    #     "--release",
    #     required=False,
    #     help="Weights release to use (default: the model's default release).",
    # )
    parser.add_argument(
        "-w",
        "--weights-dir",
        type=_existing,
        required=False,
        help="Use weights from this directory instead of downloading a release.",
    )
    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        default=False,
        help="Log detailed informations.",
    )
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    if args.model == "nnunet":
        try:
            from cochlea_detector.infer_nnunet import infer  # noqa: PLC0415
        except ImportError as e:
            print(
                f"{parser.prog}: error: the 'nnunet' model needs the optional "
                f"nnunetv2 dependency. Install it with "
                f"`pip install -e '.[nnunet]'` or `uv sync --extra nnunet` ({e}).",
                file=sys.stderr,
            )
            return 1
    else:
        from cochlea_detector.infer_yolo import infer  # noqa: PLC0415

    try:
        infer(
            args.input,
            args.output,
            device=args.device,
            release=args.release,
            weights_dir=args.weights_dir,
        )
    except (ValueError, NotImplementedError, RuntimeError, OSError) as exc:
        if args.verbose:
            logger.exception("Inference failed")
        else:
            print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
