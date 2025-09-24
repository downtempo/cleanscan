# cleanscan
Scan audio files for specific words and tag them in a local database if they do/don't have the words.

## Requirements

* Python 3.9+
* [ffmpeg](https://ffmpeg.org/) available on your ``PATH`` (used for audio fingerprinting)
* ``openai`` Python package configured with an ``OPENAI_API_KEY`` environment variable

Install dependencies with:

```bash
pip install openai
```

## Usage

```
python cleanscan.py /path/to/audio \
  --banned-words banned_words.txt \
  --database cleanscan.db \
  --recursive
```

The script will:

1. Generate an audio fingerprint for each MP3/AAC/M4A file (optionally scanning sub-directories).
2. Skip files that were already processed (based on the fingerprint stored in the SQLite database).
3. Call the OpenAI transcription API to generate transcripts for new files.
4. Search the transcripts for exact matches and near-misses of the banned terms.
5. Record results in the SQLite database and print a summary report.

Command line options:

* ``--recursive`` – search subdirectories.
* ``--model`` – override the OpenAI transcription model (defaults to ``gpt-4o-mini-transcribe``).
* ``--near-threshold`` – similarity threshold for near matches (defaults to ``0.85``).
* ``--verbose`` – enable debug logging output.

The report summarises how many files were scanned, how many were skipped due to existing fingerprints, and which files triggered banned word matches.
