"""Download official AIST music tracks and prepare WAV files for conditioning.

The AIST++ motion archive contains SMPL motion only. Music conditioning uses the
official AIST Dance Database music files listed in:

https://aistdancedb.ongaaccel.jp/v1.0.0/data/all_music_wav_url.csv

The CSV currently contains direct audio URLs, typically MP3 links. This script
downloads those source files and converts each track to a mono WAV named by
music id, for example ``mBR0.wav``. The motion loader maps AIST++ sequence names
such as ``gBR_sBM_cAll_d04_mBR0_ch02.pkl`` to ``mBR0.wav``.

You must read and accept the AIST Dance Database Terms of Use before using
``--agree-terms``:

https://aistdancedb.ongaaccel.jp/terms_of_use/
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import shutil
import ssl
import subprocess
from urllib.parse import urlparse
import urllib.request


MUSIC_CSV_URL = "https://aistdancedb.ongaaccel.jp/v1.0.0/data/all_music_wav_url.csv"


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _ffmpeg_binary() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _download(url: str, target: Path, overwrite: bool = False) -> None:
    """Download a URL with local caching."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        logging.info("Using cached source audio: %s", target)
        return
    logging.info("Downloading %s", url)
    tmp_target = target.with_suffix(target.suffix + ".part")
    if tmp_target.exists():
        tmp_target.unlink()
    curl_cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "-A",
        "Mozilla/5.0",
        "-o",
        str(tmp_target),
        url,
    ]
    completed = subprocess.run(curl_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        context = ssl._create_unverified_context()
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, context=context, timeout=180) as response, tmp_target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    tmp_target.replace(target)
    logging.info("Saved %s", target)


def _download_text(url: str, target: Path, overwrite: bool = False) -> None:
    """Download a text file with local caching."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        logging.info("Using cached CSV: %s", target)
        return
    logging.info("Downloading %s", url)
    context = ssl._create_unverified_context()
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, context=context, timeout=60) as response:
        target.write_bytes(response.read())
    logging.info("Saved %s", target)


def _parse_audio_urls(csv_path: Path) -> list[str]:
    """Parse AIST audio URLs from a one-column or generic CSV file."""
    urls: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            for cell in row:
                value = cell.strip()
                if value.startswith("http://") or value.startswith("https://"):
                    urls.append(value)
                    break
    unique_urls = sorted(dict.fromkeys(urls))
    if not unique_urls:
        raise ValueError(f"No audio URLs found in {csv_path}")
    return unique_urls


def _source_filename(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise ValueError(f"Could not infer filename from URL: {url}")
    return name


def _convert_to_wav(source_path: Path, wav_path: Path, sample_rate: int, overwrite: bool = False) -> None:
    """Convert an audio file to mono WAV for deterministic librosa loading."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if wav_path.exists() and wav_path.stat().st_size > 0 and not overwrite:
        logging.info("Using existing WAV: %s", wav_path)
        return
    cmd = [
        _ffmpeg_binary(),
        "-y" if overwrite else "-n",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(wav_path),
    ]
    logging.info("Converting %s -> %s", source_path.name, wav_path.name)
    completed = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {source_path}: {completed.stderr[-1200:]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official AIST music files and convert them to WAV.")
    parser.add_argument("--csv-url", type=str, default=MUSIC_CSV_URL)
    parser.add_argument("--csv-path", type=Path, default=Path("data/all_music_wav_url.csv"))
    parser.add_argument("--audio-root", type=Path, default=Path("data/aist-smpl-clean/audio"))
    parser.add_argument("--source-cache", type=Path, default=Path("data/.cache/aist_plusplus/music"))
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download-csv", action="store_true", help="Refresh the local CSV before downloading audio.")
    parser.add_argument(
        "--agree-terms",
        action="store_true",
        help="Required after you have read and accepted the AIST Dance Database Terms of Use.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if not args.agree_terms:
        raise SystemExit(
            "Read https://aistdancedb.ongaaccel.jp/terms_of_use/ and rerun with --agree-terms if you accept."
        )

    root = _project_root()
    csv_path = (root / args.csv_path).resolve()
    audio_root = (root / args.audio_root).resolve()
    source_cache = (root / args.source_cache).resolve()

    if args.download_csv or not csv_path.exists():
        _download_text(args.csv_url, csv_path, overwrite=args.overwrite)

    urls = _parse_audio_urls(csv_path)
    if args.max_files is not None:
        urls = urls[: args.max_files]
    logging.info("Preparing %d AIST music track(s).", len(urls))

    for url in urls:
        source_name = _source_filename(url)
        source_path = source_cache / source_name
        wav_path = audio_root / f"{Path(source_name).stem}.wav"
        _download(url, source_path, overwrite=args.overwrite)
        _convert_to_wav(source_path, wav_path, sample_rate=args.sample_rate, overwrite=args.overwrite)

    wav_count = len(list(audio_root.glob("m*.wav")))
    logging.info("AIST music WAV files ready: %d in %s", wav_count, audio_root)
    logging.info("Set AISTPP_AUDIO_ROOT=%s", audio_root)


if __name__ == "__main__":
    main()
