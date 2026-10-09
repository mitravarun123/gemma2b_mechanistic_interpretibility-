# podcast_trascript

Technical transcript glossary prototype

Turn a YouTube video or technical podcast into a timestamped transcript and a JSON glossary of new technological terms.

The main entry point is `python -m transcript_prototype.main`.

## Setup

Python 3.10+:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r transcript_prototype/requirements.txt
```

Install and start [Ollama](https://ollama.com), then download a local model:

```sh
ollama pull llama3.2
```

## Run

From this project directory:

```sh
# YouTube captions (English by default)
python -m transcript_prototype.main 'https://www.youtube.com/watch?v=VIDEO_ID' --output output/video.json

# Local podcast recording: MP3, WAV, M4A, or video supported by Whisper
python -m transcript_prototype.main /path/to/podcast.mp3 --output output/podcast.json

# Existing transcript, useful for trying the prototype without transcription
python -m transcript_prototype.main transcript_prototype/examples/podcast.txt --output output/demo.json
```

YouTube URLs support watch, shorts, live, embed, and youtu.be formats. This uses available captions; videos without accessible captions produce an error. For those videos, supply a local recording instead. Podcast inputs are local audio/video files; podcast webpage URLs and RSS feeds are not supported. Whisper downloads its model on first use and transcribes on CPU. Use `--language hi` for Hindi or another supported language code; explanations and Wikipedia lookup currently target English.

`--model`, `--endpoint`, and `--whisper-model` configure the local models. Long transcripts are analyzed in overlapping chunks. Ollama must be running, with the selected model installed. No paid API key is required.

## JSON output

Each episode contains `source`, `transcript` segments with timestamps, `new_terms`, and `known_terms`. Every new term includes its explanation, transcript occurrences, source URLs when found, and `information_status`:

- `reference_summary`: summary retrieved from a matching Wikipedia article.
- `model_generated_unverified`: model explanation when no matching reference is available; this has no invented source URL.

A Wikipedia reference is useful provenance, but does not independently establish correctness or resolve every ambiguous term. Inspect important definitions. Detection is model based and may miss terms. A term is considered **new** when its case-insensitive spelling is absent from `output/glossary.json`; synonyms and acronyms are not automatically merged. Choose a separate `--glossary` path for a different user's vocabulary. Runs are sequential; concurrent writes to the same glossary are not supported.

The transcript is saved before term analysis. If Ollama fails, the episode remains with `status: transcribed`; rerun to complete analysis. Completed runs update the glossary, so repeated terms appear in `known_terms`. No terms are fabricated to fill an empty result.

## Verification

```sh
python3 -m unittest transcript_prototype.test_main -v
```

Tests cover URL parsing, transcript evidence, chunking, reference fallback, persistent deduplication, and saving transcripts after analysis failure using mocked services. Live YouTube, Whisper, Wikipedia, and Ollama require your network/model setup and are not claimed verified by these tests.
