"""Audio scanning tool to detect banned words using OpenAI transcription.

This module can be executed as a script.  It will walk a directory of audio
files, generate robust audio fingerprints to avoid double processing, call the
OpenAI transcription API, and then scan the returned transcript for exact and
near matches of banned words that are provided in a separate text file.

The script stores processing results in a SQLite database so future runs can
skip audio that has already been checked.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sqlite3
import subprocess
from array import array
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from openai import OpenAI


# Supported audio file extensions (all compared in lower case)
SUPPORTED_EXTENSIONS = {".mp3", ".aac", ".m4a"}


@dataclass
class ScanResult:
    """Container for the transcription results of a single audio file."""

    file_path: Path
    fingerprint: str
    transcript: str
    matches: List[str]
    near_matches: List[Dict[str, object]]

    @property
    def is_flagged(self) -> bool:
        return bool(self.matches or self.near_matches)


def load_banned_words(banned_file: Path) -> List[str]:
    """Load banned words/phrases from a text file.

    Blank lines and lines starting with ``#`` are ignored.
    """

    words: List[str] = []
    with banned_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            words.append(cleaned)
    if not words:
        raise ValueError(f"No banned words found in {banned_file}")
    return words


def iter_audio_files(directory: Path, recursive: bool) -> Iterator[Path]:
    """Yield supported audio files from ``directory``.

    Args:
        directory: The root directory to search.
        recursive: If ``True``, descend into subdirectories.
    """

    if recursive:
        iterator: Iterable[Path] = directory.rglob("*")
    else:
        iterator = directory.glob("*")

    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def generate_audio_fingerprint(audio_path: Path, sample_rate: int = 8000, chunk_size: int = 4096) -> str:
    """Generate a robust audio fingerprint for ``audio_path``.

    The fingerprint is derived from a mono, down-sampled representation of the
    audio so that different encodings of the same source map to the same
    fingerprint.  The energy-based feature representation allows similar takes
    to hash differently (for example, a live version versus a studio version).
    """

    command = [
        "ffmpeg",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]

    logging.debug("Generating fingerprint via ffmpeg: %s", " ".join(command))
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required to compute audio fingerprints but was not found in PATH"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed to process {audio_path}: {exc.stderr.decode(errors='ignore')}" ) from exc

    pcm_data = completed.stdout
    if not pcm_data:
        raise RuntimeError(f"No PCM data produced for {audio_path}")

    samples = array("h")
    samples.frombytes(pcm_data)
    if not samples:
        raise RuntimeError(f"Unable to decode samples for {audio_path}")

    # Compute coarse energy signatures for consecutive chunks.  Quantising the
    # energies makes the fingerprint resilient to small variations while still
    # differentiating distinct performances.
    chunk_features: List[str] = []
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start : start + chunk_size]
        if not chunk:
            continue
        energy = sum(int(sample) * int(sample) for sample in chunk) / len(chunk)
        quantised = int(energy // 1_000)
        chunk_features.append(str(quantised))

    feature_string = ",".join(chunk_features)
    # Local import to avoid unnecessary dependency during module import.
    import hashlib

    return hashlib.sha256(feature_string.encode("utf-8")).hexdigest()


def ensure_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_files (
            fingerprint TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            checked_at TEXT NOT NULL,
            transcript TEXT,
            matches TEXT,
            near_matches TEXT
        )
        """
    )
    connection.commit()


def is_already_processed(connection: sqlite3.Connection, fingerprint: str) -> bool:
    cursor = connection.execute(
        "SELECT 1 FROM processed_files WHERE fingerprint = ?", (fingerprint,)
    )
    return cursor.fetchone() is not None


def store_result(connection: sqlite3.Connection, result: ScanResult) -> None:
    connection.execute(
        """
        INSERT OR REPLACE INTO processed_files (
            fingerprint,
            file_path,
            checked_at,
            transcript,
            matches,
            near_matches
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            result.fingerprint,
            str(result.file_path),
            dt.datetime.now(dt.timezone.utc).isoformat(),
            result.transcript,
            json.dumps(result.matches, ensure_ascii=False),
            json.dumps(result.near_matches, ensure_ascii=False),
        ),
    )
    connection.commit()


def transcribe_audio(client: OpenAI, audio_path: Path, model: str) -> str:
    with audio_path.open("rb") as audio_file:
        logging.info("Transcribing %s", audio_path)
        response = client.audio.transcriptions.create(model=model, file=audio_file)
    return response.text


def normalise_text(text: str) -> Tuple[str, List[str], str]:
    import re

    lowered = text.lower()
    tokens = re.findall(r"\b[\w']+\b", lowered)
    token_text = " ".join(tokens)
    return lowered, tokens, token_text


def find_matches(
    transcript: str,
    banned_words: Sequence[str],
    near_match_threshold: float,
) -> Tuple[List[str], List[Dict[str, object]]]:
    """Return exact matches and near matches for ``banned_words`` in ``transcript``."""

    _, tokens, tokenised_transcript = normalise_text(transcript)
    exact_matches: List[str] = []
    near_matches: List[Dict[str, object]] = []

    for banned in banned_words:
        _, banned_tokens, banned_phrase = normalise_text(banned)
        if not banned_phrase:
            continue

        if banned_phrase in tokenised_transcript:
            exact_matches.append(banned)
            continue

        n = len(banned_tokens)
        if n == 0 or not tokens:
            continue

        for index in range(0, len(tokens) - n + 1):
            window = " ".join(tokens[index : index + n])
            ratio = SequenceMatcher(None, window, banned_phrase).ratio()
            if ratio >= near_match_threshold:
                near_matches.append(
                    {
                        "banned_word": banned,
                        "matched_text": window,
                        "similarity": round(ratio, 3),
                    }
                )
                break

    return exact_matches, near_matches


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan audio files for banned words using OpenAI transcripts.")
    parser.add_argument("directory", type=Path, help="Directory containing audio files to scan.")
    parser.add_argument(
        "--banned-words",
        type=Path,
        required=True,
        help="Path to a text file containing banned words or phrases (one per line).",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("cleanscan.db"),
        help="SQLite database used to store processed fingerprints (default: cleanscan.db).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini-transcribe",
        help="OpenAI transcription model to use (default: gpt-4o-mini-transcribe).",
    )
    parser.add_argument(
        "--near-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold (0-1) for near-miss detection (default: 0.85).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for audio files inside the provided directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser


def format_report(new_results: Sequence[ScanResult], total_files: int, skipped: int) -> str:
    processed_count = len(new_results)
    flagged = sum(1 for result in new_results if result.is_flagged)
    lines = [
        "Scan complete.",
        f"Total audio files discovered: {total_files}",
        f"Already processed (skipped): {skipped}",
        f"Processed via OpenAI API: {processed_count}",
        f"Files flagged (exact or near matches): {flagged}",
    ]

    if new_results:
        lines.append("")
        lines.append("Detailed results:")
        for result in new_results:
            status = "FLAGGED" if result.is_flagged else "clear"
            lines.append(f"- {result.file_path} [{status}]")
            if result.matches:
                lines.append(f"    Exact matches: {', '.join(result.matches)}")
            if result.near_matches:
                for near in result.near_matches:
                    lines.append(
                        "    Near match: '{matched_text}' ~ '{banned_word}' (similarity {similarity})".format(
                            **near
                        )
                    )

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.directory.exists() or not args.directory.is_dir():
        parser.error(f"Directory {args.directory} does not exist or is not a directory")

    if not args.banned_words.exists():
        parser.error(f"Banned words file {args.banned_words} does not exist")

    banned_words = load_banned_words(args.banned_words)
    logging.info("Loaded %d banned terms", len(banned_words))

    connection = sqlite3.connect(args.database)
    ensure_database(connection)

    client = OpenAI()

    total_files = 0
    skipped = 0
    new_results: List[ScanResult] = []

    for audio_path in iter_audio_files(args.directory, args.recursive):
        total_files += 1
        logging.debug("Processing %s", audio_path)
        try:
            fingerprint = generate_audio_fingerprint(audio_path)
        except RuntimeError as exc:
            logging.error("Failed to fingerprint %s: %s", audio_path, exc)
            continue

        if is_already_processed(connection, fingerprint):
            logging.info("Skipping %s (already processed)", audio_path)
            skipped += 1
            continue

        try:
            transcript_text = transcribe_audio(client, audio_path, args.model)
        except Exception as exc:  # Broad except to surface API errors but continue.
            logging.error("Failed to transcribe %s: %s", audio_path, exc)
            continue

        matches, near_matches = find_matches(transcript_text, banned_words, args.near_threshold)
        result = ScanResult(
            file_path=audio_path,
            fingerprint=fingerprint,
            transcript=transcript_text,
            matches=matches,
            near_matches=near_matches,
        )

        store_result(connection, result)
        new_results.append(result)

    report = format_report(new_results, total_files, skipped)
    print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
