"""Download and organize official AIST++ motion data for local development.

This script uses official Google-hosted AIST++ resources:
    - downloader.py from google/aistplusplus_api
    - motions.zip from Google Cloud Storage
    - splits.zip from Google Cloud Storage
    - ignore_list.txt from google/aistplusplus_api

Outputs are organized relative to the project root:
    data/aist_plusplus/motions/
    data/aist_plusplus/splits/
    data/aist_plusplus/ignore/ignore_list.txt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import urllib.request
import zipfile


DOWNLOADER_URL = "https://raw.githubusercontent.com/google/aistplusplus_api/main/downloader.py"
MOTIONS_URL = "https://storage.googleapis.com/aist_plusplus/motions.zip"
SPLITS_URL = "https://storage.googleapis.com/aist_plusplus/splits.zip"
IGNORE_LIST_URL = "https://raw.githubusercontent.com/google/aistplusplus_api/main/ignore_list.txt"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _download(url: str, target: Path, overwrite: bool = False) -> None:
    """Download a URL if target is absent or overwrite is requested."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        logging.info("Using existing file: %s", target)
        return
    logging.info("Downloading %s", url)
    urllib.request.urlretrieve(url, target)
    logging.info("Saved %s", target)


def _extract_zip_flat(zip_path: Path, output_dir: Path, suffixes: tuple[str, ...] | None = None) -> int:
    """Extract files from a zip into output_dir using flat filenames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            source_name = Path(member).name
            if not source_name:
                continue
            if suffixes is not None and not source_name.lower().endswith(suffixes):
                continue
            target = output_dir / source_name
            if not target.exists():
                with archive.open(member) as source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
            count += 1
    return count


def _copy_existing_clean_dataset(clean_root: Path, output_root: Path) -> bool:
    """Copy already-downloaded local clean dataset into canonical layout."""
    motions = clean_root / "motions"
    splits = clean_root / "splits"
    ignore_list = clean_root / "ignore_list.txt"
    if not motions.exists() or not splits.exists() or not ignore_list.exists():
        return False

    output_motions = output_root / "motions"
    output_splits = output_root / "splits"
    output_ignore = output_root / "ignore" / "ignore_list.txt"
    output_motions.mkdir(parents=True, exist_ok=True)
    output_splits.mkdir(parents=True, exist_ok=True)
    output_ignore.parent.mkdir(parents=True, exist_ok=True)

    for path in motions.glob("*.pkl"):
        target = output_motions / path.name
        if not target.exists():
            shutil.copy2(path, target)
    for path in splits.glob("*"):
        if path.is_file():
            target = output_splits / path.name
            if not target.exists():
                shutil.copy2(path, target)
    if not output_ignore.exists():
        shutil.copy2(ignore_list, output_ignore)

    logging.info("Copied existing clean AIST++ dataset from %s to %s", clean_root, output_root)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official AIST++ motion data and splits.")
    parser.add_argument("--output-root", type=Path, default=Path("data/aist_plusplus"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.cache/aist_plusplus"))
    parser.add_argument("--existing-clean-root", type=Path, default=Path("data/aist-smpl-clean"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--prefer-existing",
        action="store_true",
        default=True,
        help="Use data/aist-smpl-clean if already downloaded locally.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    root = _project_root()
    output_root = (root / args.output_root).resolve()
    cache_dir = (root / args.cache_dir).resolve()
    existing_clean_root = (root / args.existing_clean_root).resolve()

    if args.prefer_existing and not args.overwrite:
        if _copy_existing_clean_dataset(existing_clean_root, output_root):
            logging.info("Canonical AIST++ layout ready: %s", output_root)
            logging.info("AISTPP_ROOT=%s", output_root / "motions")
            logging.info("AISTPP_SPLIT_ROOT=%s", output_root / "splits")
            logging.info("AISTPP_IGNORE_LIST=%s", output_root / "ignore" / "ignore_list.txt")
            return

    downloader_path = cache_dir / "downloader.py"
    motions_zip = cache_dir / "motions.zip"
    splits_zip = cache_dir / "splits.zip"
    ignore_list_path = output_root / "ignore" / "ignore_list.txt"

    _download(DOWNLOADER_URL, downloader_path, overwrite=args.overwrite)
    _download(MOTIONS_URL, motions_zip, overwrite=args.overwrite)
    _download(SPLITS_URL, splits_zip, overwrite=args.overwrite)
    _download(IGNORE_LIST_URL, ignore_list_path, overwrite=args.overwrite)

    motion_count = _extract_zip_flat(motions_zip, output_root / "motions", suffixes=(".pkl",))
    split_count = _extract_zip_flat(splits_zip, output_root / "splits")

    logging.info("Official AIST++ data ready.")
    logging.info("Motion files prepared: %d", motion_count)
    logging.info("Split files prepared: %d", split_count)
    logging.info("Official downloader cached: %s", downloader_path)
    logging.info("AISTPP_ROOT=%s", output_root / "motions")
    logging.info("AISTPP_SPLIT_ROOT=%s", output_root / "splits")
    logging.info("AISTPP_IGNORE_LIST=%s", ignore_list_path)


if __name__ == "__main__":
    main()
