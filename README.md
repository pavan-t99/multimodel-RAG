# multimodel-RAG


# Ayushman Bharath (PM-JAY) Multilingual Voice RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions about registering for and using the **Ayushman Bharath / PM-JAY** health insurance scheme — in **English, Hindi, Telugu, or Nepali**, by **text or voice**.

The knowledge base is built from official PM-JAY PDFs (registration guides, process flows, preauthorization forms, TMS manuals). The RAG pipeline itself always runs in English; a translation layer sits at the edges so users can ask and receive answers in their own language, spoken or typed.

## How it works

```
 audio in ──▶ ASR (faster-whisper) ──▶ native text
                                            │
 typed text ────────────────────────────────┤
                                            ▼
                              translation.to_english()  (IndicTrans2)
                                            │
                                            ▼
                              HyDE retrieval over FAISS index
                                            │
                                            ▼
                              Groq (Llama 3.3 70B) generates answer
                                            │
                                            ▼
                              translation.from_english()  (IndicTrans2)
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                            ▼
                        typed reply                 TTS (MMS-TTS) ──▶ spoken reply
```

**Retrieval — HyDE (Hypothetical Document Embeddings).** Instead of embedding the raw user query, `HyDERetriever` (in `models.py`) first asks the LLM to generate a plausible, textbook-style answer to the question, then embeds *that* and searches the FAISS index with it. This tends to match dense document chunks better than short user queries do. Answers with a low relevance score are treated as "no match" and the bot responds that it couldn't find an answer, instead of hallucinating.

**Generation.** The retrieved chunks + recent conversation history are assembled into a prompt (`Rag_pipeline.py`) with rules that keep answers short, in bullet points, English-only-at-this-stage, grounded strictly in the retrieved context, and with any links formatted as clickable HTML anchors.

**Translation.** `translation.py` wraps AI4Bharat's **IndicTrans2** (distilled 200M checkpoints, CPU-friendly) with two directional models — indic→English and English→indic — loaded once and reused per request. URLs and `<a href>` links are protected from translation so they survive round-trip intact. Nepali translation quality is noted as weaker than Hindi/Telugu due to sparser AI4Bharat training data for that direction.

**Speech.** `speech.py` adds voice input/output on top of the text pipeline:
- **ASR:** `faster-whisper` ("small", int8, CPU) transcribes mic or uploaded audio (wav/flac/ogg/opus/mp3/m4a/aac) into native-language text. Audio is always decoded via `pydub`/ffmpeg and resampled to 16kHz mono before transcription.
- **TTS:** Meta's MMS-TTS (VITS), one checkpoint per language, synthesizes the (already-translated) reply back into speech.
- Both are CPU-first by default, with commented-out GPU config (`device="cuda"`, larger Whisper checkpoint, fp16) ready to enable if a GPU is available.

**Index building.** `build_index.py` walks a PDF dataset (currently pointed at a Kaggle input path), extracts text per file with `PyMuPDFLoader`, and OCRs embedded images via `RapidOCRBlobParser` — except for very large files (>12MB or filenames containing "operational_guidelines"), where OCR is skipped to avoid memory blowups and only native text is read. Chunks (800 chars, 100 overlap) are embedded with `all-MiniLM-L6-v2` and stored as a local FAISS index (`data/faiss_index/`). `models.py` will auto-trigger a rebuild by calling this script if the index directory is missing, or if `FORCE_REBUILD_INDEX=True` is set.

## Project structure

```
app.py                Streamlit UI: language picker, chat, mic input, audio upload, TTS playback
Rag_pipeline.py        Orchestrates HyDE retrieval + prompt assembly + LLM call; get_answer() entry point
models.py               LLM init (Groq), embedding model, FAISS index loader, HyDERetriever
translation.py          IndicTrans2 wrapper: to_english() / from_english() / warm_up()
speech.py                faster-whisper ASR + MMS-TTS, audio decoding utilities
build_index.py           Builds the FAISS index from the PM-JAY PDF dataset
req_res.py               Simple Request/Response data classes used across the pipeline
listmodels.py            Standalone script to list available Gemini models for a Google API key
PM_JAY/                  Sample PM-JAY source PDFs (registration, process flow, forms, manuals)
data/faiss_index/        Persisted FAISS vector store (generated, not hand-authored)
requirements.txt         Python dependencies
```

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You'll also need **ffmpeg** installed on the system (required by `pydub` for audio decoding in `speech.py`).

### 2. Configure environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
CHATTING_GROQ_API_KEY=your_second_groq_api_key_here
```

- `GROQ_API_KEY` — used for the main answer-generation LLM (`models.init_llm_model`).
- `CHATTING_GROQ_API_KEY` — used by the HyDE hypothetical-document generator (`HyDERetriever`). Kept separate so the two Groq calls can use different rate-limit quotas / keys.

Both are required — the app raises on startup if either is missing.

> **Security note:** `listmodels.py` currently has a Google API key hardcoded in the source. Move it to an environment variable (e.g. `GOOGLE_API_KEY`) and revoke/rotate the existing key before pushing this repo anywhere public.

### 3. Build the vector index

`build_index.py` is currently configured to read PDFs from a Kaggle dataset path (`/kaggle/input/datasets/pavankurman/pm-jay-registrations/PM_JAY_REGISTRATION`). To index the PDFs shipped in `PM_JAY/` instead, update `dataset_path` in `build_index.py` to point there, then run:

```bash
python build_index.py
```

This produces `data/faiss_index/`. If it doesn't exist yet, the app will build it automatically on first run; force a rebuild anytime with:

```bash
FORCE_REBUILD_INDEX=True python app.py
```

### 4. Run the app

```bash
streamlit run app.py
```

On first launch, translation and speech models are downloaded from Hugging Face and warmed up (~10–20s) — subsequent runs use the local cache.

## Using the chatbot

1. Pick a language (English / Hindi / Telugu / Nepali) from the dropdown.
2. Ask a question by:
   - typing it in the chat box,
   - recording it live via the mic, or
   - uploading an audio file (useful in environments like Codespaces where live mic access is blocked).
3. The bot retrieves relevant PM-JAY context, generates an answer strictly grounded in that context, translates it back to your chosen language if needed, and plays it back as speech in addition to showing it as text.
4. If no sufficiently relevant context is found for the query, the bot responds with a fixed "please enter a valid PM-JAY-related question" message rather than guessing.

## Key design choices worth knowing

- **English-only knowledge base, multilingual edges.** Retrieval and generation always happen in English; translation is applied only to the user-facing query and answer. This avoids needing separate embeddings/indexes per language.
- **HyDE over raw-query retrieval** for better semantic match against long-form document chunks.
- **Relevance-score gating** (`score < 0.5`) prevents the bot from answering when nothing in the index is actually relevant.
- **CPU-first everywhere** (translation, ASR, TTS) so the whole stack runs in a plain container/Codespaces without a GPU; GPU code paths are present but commented out.
- **Link-safe translation**: hyperlinks are stashed out before translation and restored afterward so IndicTrans2 doesn't corrupt URLs/markup.

## Known gaps / things to check before relying on this

- `dataset_path` in `build_index.py` is hardcoded to a Kaggle path — needs updating for local/other environments.
- Nepali translation quality is explicitly weaker per AI4Bharat's own training data coverage — validate against real questions before trusting it.
- No automated tests currently exist for retrieval quality, translation accuracy, or the ASR/TTS round-trip.
- The hardcoded API key in `listmodels.py` (see Security note above).
