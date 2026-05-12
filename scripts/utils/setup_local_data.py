"""Download a toy AIST++ subset for local Mac debugging.

The script downloads the official Google-hosted AIST++ motion and split archives,
extracts only the first N SMPL motion pickle files, and prepares a small local
directory tree compatible with configs/testing.yaml.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import urllib.request
import zipfile


MOTIONS_URL = "https://storage.googleapis.com/aist_plusplus/motions.zip"
SPLITS_URL = "https://storage.googleapis.com/aist_plusplus/splits.zip"


def _download(url: str, destination: Path) -> None:
    """Download url to destination if the file is absent."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        logging.info("Using existing archive: %s", destination)
        return
    logging.info("Downloading %s", url)
    urllib.request.urlretrieve(url, destination)
    logging.info("Saved archive: %s", destination)


def _extract_first_motion_files(archive_path: Path, output_dir: Path, max_files: int) -> list[Path]:
    """Extract the first max_files .pkl motion files from motions.zip."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        motion_members = sorted(member for member in archive.namelist() if member.lower().endswith(".pkl"))
        if not motion_members:
            raise RuntimeError(f"No .pkl motion files found in {archive_path}")

        for member in motion_members[:max_files]:
            target = output_dir / Path(member).name
            if not target.exists():
                with archive.open(member) as source, target.open("wb") as handle:
                    handle.write(source.read())
            extracted.append(target)
            logging.info("Prepared toy motion file: %s", target)
    return extracted


def _extract_splits(archive_path: Path, output_dir: Path) -> list[Path]:
    """Extract all official split files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            target = output_dir / Path(member).name
            if not target.exists():
                with archive.open(member) as source, target.open("wb") as handle:
                    handle.write(source.read())
            extracted.append(target)
            logging.info("Prepared split file: %s", target)
    return extracted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a toy AIST++ subset for local debugging.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/aist_plusplus"))
    parser.add_argument("--max-motion-files", type=int, default=5)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.cache"))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if args.max_motion_files <= 0:
        raise ValueError("--max-motion-files must be positive")

    motions_archive = args.cache_dir / "motions.zip"
    splits_archive = args.cache_dir / "splits.zip"
    _download(MOTIONS_URL, motions_archive)
    _download(SPLITS_URL, splits_archive)

    motion_dir = args.output_dir / "motions"
    split_dir = args.output_dir / "splits"
    motion_files = _extract_first_motion_files(motions_archive, motion_dir, args.max_motion_files)
    split_files = _extract_splits(splits_archive, split_dir)

    logging.info("Toy AIST++ setup complete.")
    logging.info("Motion files: %d", len(motion_files))
    logging.info("Split files: %d", len(split_files))
    logging.info("Use: export AISTPP_ROOT=%s", motion_dir.resolve())
    logging.info("Use: export AISTPP_SPLIT_ROOT=%s", split_dir.resolve())


if __name__ == "__main__":
    main()
