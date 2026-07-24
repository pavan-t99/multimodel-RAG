"""
Speech layer for the Ayushman Bharath chatbot: mic -> text (ASR) and
text -> spoken audio (TTS), sitting on the outer edges of the existing
translation + RAG pipeline. Flow becomes:

  mic audio --[transcribe]--> native text --[translation.to_english]--> English
    --[RAG/LLM]--> English answer --[translation.from_english]--> native text
    --[synthesize]--> spoken audio reply

CPU-first design (this file's current settings):
  - ASR: faster-whisper, "small" multilingual checkpoint, int8 quantized.
    Much faster than plain openai-whisper on CPU while still covering
    Hindi/Telugu/Nepali/English.
  - TTS: Meta's MMS-TTS (VITS), one small single-speaker checkpoint per
    language, loaded via plain `transformers`. CPU inference is fine for
    single-utterance synthesis.

GPU upgrade path (commented inline below -- flip these on if you get a GPU):
  - faster-whisper: device="cuda", compute_type="float16" (or
    "int8_float16" for a speed/memory middle ground), and bump the model
    size up to "medium" or "large-v3" for meaningfully better accuracy.
  - MMS-TTS: model.to("cuda"); mainly helps if you batch multiple
    synthesis requests rather than single-utterance replies.
"""

import io
import re

import torch
import soundfile as sf
from faster_whisper import WhisperModel
from transformers import VitsModel, AutoTokenizer

DEVICE = "cpu"
# GPU: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- ASR (speech -> text) ----------------

# --- CPU config (current) ---
_WHISPER_MODEL_SIZE = "small"
_WHISPER_COMPUTE_TYPE = "int8"
# --- GPU config (uncomment when you have a GPU, comment out the two above) ---
# _WHISPER_MODEL_SIZE = "large-v3"
# _WHISPER_COMPUTE_TYPE = "float16"

_whisper = None


def _get_whisper():
    global _whisper
    if _whisper is None:
        print(
            f"Loading faster-whisper ({_WHISPER_MODEL_SIZE}, "
            f"{_WHISPER_COMPUTE_TYPE}) on {DEVICE} ..."
        )
        _whisper = WhisperModel(
            _WHISPER_MODEL_SIZE,
            device=DEVICE,  # GPU: device="cuda"
            compute_type=_WHISPER_COMPUTE_TYPE,
        )
    return _whisper


# Whisper wants ISO-639-1 (2-letter) language hints, not our FLORES-style
# codes from translation.py.
WHISPER_LANG_HINTS = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Nepali": "ne",
}


def transcribe(audio_bytes: bytes, language: str) -> str:
    """Transcribe recorded mic audio (wav bytes) into text, in the user's
    selected language. Returns the transcript still in that native
    language -- translation.to_english() handles the rest downstream,
    same as typed input does."""
    model = _get_whisper()
    audio_array, _sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio_array.ndim > 1:  # collapse stereo to mono if needed
        audio_array = audio_array.mean(axis=1)

    lang_hint = WHISPER_LANG_HINTS.get(language)  # None -> auto-detect
    segments, _info = model.transcribe(audio_array, language=lang_hint, beam_size=5)
    return " ".join(seg.text.strip() for seg in segments).strip()


# ---------------- TTS (text -> speech) ----------------

# Meta MMS-TTS checkpoints, one per language.
MMS_TTS_MODELS = {
    "English": "facebook/mms-tts-eng",
    "Hindi": "facebook/mms-tts-hin",
    "Telugu": "facebook/mms-tts-tel",
    "Nepali": "facebook/mms-tts-npi",
}

_tts_cache = {}


def _get_tts(language: str):
    if language not in _tts_cache:
        model_name = MMS_TTS_MODELS[language]
        print(f"Loading TTS model for {language}: {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = VitsModel.from_pretrained(model_name).to(DEVICE)  # GPU: .to("cuda")
        model.eval()
        _tts_cache[language] = (tokenizer, model)
    return _tts_cache[language]


_HTML_TAG = re.compile(r"<[^>]+>")
_BULLET_PREFIX = re.compile(r"^[\*\-]\s*", re.MULTILINE)


def clean_for_speech(text: str) -> str:
    """Strip HTML links and markdown bullet markers so the TTS model
    doesn't read out raw markup -- e.g. the <a href=...> tags the RAG
    system prompt generates, or leading '* ' bullets."""
    text = _HTML_TAG.sub("", text)
    text = _BULLET_PREFIX.sub("", text)
    return text.strip()


def synthesize(text: str, language: str) -> bytes:
    """Turn already-translated, native-language answer text into
    playable WAV audio bytes."""
    tokenizer, model = _get_tts(language)
    inputs = tokenizer(clean_for_speech(text), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model(**inputs).waveform
    waveform = output.squeeze().cpu().numpy()

    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate=model.config.sampling_rate, format="WAV")
    return buffer.getvalue()


def warm_up_speech():
    """Optional: preload ASR at startup. TTS models load lazily per
    language on first use (only 4 languages, cheap either way)."""
    _get_whisper()