"""
Translation layer for the Ayushman Bharath chatbot.

Wraps AI4Bharat's IndicTrans2 (distilled 200M checkpoints) to translate:
  - user query: native language -> English   (before RAG retrieval/generation)
  - bot answer: English -> native language   (after RAG generation)

Why IndicTrans2 distilled-200M specifically:
  - Best-benchmarked free model for Telugu/Hindi/Nepali (outperforms NLLB and
    GPT-3.5 on Indic directions per AI4Bharat's published evals).
  - 200M distilled checkpoints are the deployable size for CPU-only
    environments like a default GitHub Codespaces container (no GPU).
  - Two separate directional models (indic->en, en->indic) rather than one
    bidirectional model, so we load both once and reuse them per request.

Nepali caveat: AI4Bharat's own data pipeline had sparser alignment data for
Nepali than Telugu/Hindi, so expect Nepali translation quality to lag a bit.
Validate against real PM-JAY questions before trusting it in production.
"""

import re
from functools import lru_cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# --- Compatibility shim ---
# IndicTransToolkit's collator.py does:
#   from transformers.tokenization_utils import PreTrainedTokenizerBase
# Newer transformers releases dropped that re-export (it now lives only in
# transformers.tokenization_utils_base). Patch it back in before importing
# the toolkit, so this works even if requirements.txt pinning gets skipped
# or a newer transformers is already cached in the venv.
import transformers.tokenization_utils as _tu
if not hasattr(_tu, "PreTrainedTokenizerBase"):
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB
    _tu.PreTrainedTokenizerBase = _PTB

from IndicTransToolkit.processor import IndicProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FLORES-200 style language codes IndicTrans2 expects
LANG_CODES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Telugu": "tel_Telu",
    "Nepali": "npi_Deva",
}

INDIC_EN_MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"
EN_INDIC_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"

_ip = IndicProcessor(inference=True)

# Protects URLs and the <a href=...>...</a> hyperlinks the system prompt asks
# the LLM to produce, so translation doesn't mangle markup/links.
_LINK_PATTERN = re.compile(
    r"<a[^>]*>.*?</a>"          # HTML anchors
    r"|\[[^\]]*\]\([^\)]*\)"    # Markdown links: [text](url "title")
    r"|https?://\S+",           # bare URLs
    re.DOTALL,
)

@lru_cache(maxsize=2)
def _load(model_name: str):
    """Load + cache a model/tokenizer pair. Runs once per process."""
    print(f"Loading translation model: {model_name} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


def _translate(sentences, src_lang: str, tgt_lang: str, model_name: str):
    tokenizer, model = _load(model_name)
    batch = _ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(
        batch, truncation=True, padding="longest", return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        generated = model.generate(
            **inputs, use_cache=True, min_length=0, max_length=256, num_beams=5
        )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return _ip.postprocess_batch(decoded, lang=tgt_lang)


def _protect_links(text: str):
    links = []

    def _stash(match):
        links.append(match.group(0))
        return f" @@LINK{len(links) - 1}@@ "

    return _LINK_PATTERN.sub(_stash, text), links


def _restore_links(text: str, links: list) -> str:
    for i, link in enumerate(links):
        # IndicTrans2's tokenizer often reshapes the placeholder during
        # translation (e.g. "@@LINK0@@" -> "@@LINK0 @@"), so match loosely.
        pattern = re.compile(r"@+\s*LINK\s*" + str(i) + r"\s*@+", re.IGNORECASE)
        text = pattern.sub(link, text)
    return text


def to_english(text: str, source_language: str) -> str:
    """Translate the user's query into English before it hits retrieval/HyDE."""
    if source_language == "English" or not text.strip():
        return text
    src = LANG_CODES[source_language]
    return _translate([text], src, "eng_Latn", INDIC_EN_MODEL)[0]


def from_english(text: str, target_language: str) -> str:
    """Translate the English RAG answer back into the user's chosen language."""
    if target_language == "English" or not text.strip():
        return text
    protected, links = _protect_links(text)
    tgt = LANG_CODES[target_language]
    translated = _translate([protected], "eng_Latn", tgt, EN_INDIC_MODEL)[0]
    return _restore_links(translated, links)


def warm_up():
    """Optional: call once at app startup so the first user isn't stuck
    waiting ~10-20s for both models to load from disk/HF cache."""
    _load(INDIC_EN_MODEL)
    _load(EN_INDIC_MODEL)