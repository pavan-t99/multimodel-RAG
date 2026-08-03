import streamlit as st
from Rag_pipeline import get_answer
from speech import transcribe, synthesize, warm_up_speech

st.set_page_config(
    page_title="Ayushman Bharath Chatbot",
    page_icon="🩺",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Light styling -- kept minimal on purpose (no heavy custom theming),
# just clean spacing, a subtle card look for messages, and readable badges.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 780px; }
        h1 { margin-bottom: 0.2rem; }
        .subtitle { color: #6b7280; font-size: 1.02rem; margin-bottom: 1.4rem; }
        .section-label {
            font-weight: 600; font-size: 0.95rem; color: #374151;
            margin: 0.6rem 0 0.4rem 0;
        }
        [data-testid="stChatMessage"] { border-radius: 12px; }
        .voice-badge {
            display: inline-block; padding: 2px 10px; border-radius: 999px;
            font-size: 0.78rem; font-weight: 600; margin-left: 8px;
        }
        .badge-on  { background: #dcfce7; color: #166534; }
        .badge-off { background: #fef3c7; color: #92400e; }
    </style>
    """,
    unsafe_allow_html=True,
)

LANGUAGES = ["English", "Hindi", "Telugu", "Nepali"]
LANGUAGE_NATIVE = {
    "English": "English",
    "Hindi": "हिंदी",
    "Telugu": "తెలుగు",
    "Nepali": "नेपाली",
}
# Languages with reliable, well-supported voice (ASR + TTS) quality.
# Telugu/Nepali text translation still works fine -- only voice is limited,
# so those two languages fall back to text-only chat.
VOICE_SUPPORTED_LANGUAGES = {"English", "Hindi"}

if "language" not in st.session_state:
    st.session_state.language = "English"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_input_bytes" not in st.session_state:
    st.session_state.last_audio_input_bytes = None
if "last_uploaded_audio_bytes" not in st.session_state:
    st.session_state.last_uploaded_audio_bytes = None
if "reply_audio" not in st.session_state:
    st.session_state.reply_audio = None
if "speech_warm" not in st.session_state:
    with st.spinner("Loading voice models (first run only, ~10-20s)..."):
        warm_up_speech()
    st.session_state.speech_warm = True

# ---------------------------------------------------------------------------
# Sidebar -- language + voice status live here, always visible while scrolling
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.session_state.language = st.selectbox(
        "Choose your language",
        LANGUAGES,
        index=LANGUAGES.index(st.session_state.language),
        format_func=lambda lang: f"{lang} / {LANGUAGE_NATIVE[lang]}",
    )

    voice_enabled = st.session_state.language in VOICE_SUPPORTED_LANGUAGES
    badge_class = "badge-on" if voice_enabled else "badge-off"
    badge_text = "Voice available" if voice_enabled else "Text only"
    st.markdown(
        f"**Voice support:** <span class='voice-badge {badge_class}'>{badge_text}</span>",
        unsafe_allow_html=True,
    )

    if not voice_enabled:
        st.info(
            "🎙️ Voice input/output for Telugu and Nepali is less accurate right now.\n\n"
            "Please **type your question** instead -- you'll still get full answers "
            "in Telugu/Nepali text, just without a spoken reply.",
            icon="ℹ️",
        )

    st.divider()
    st.caption("Ayushman Bharath Assistant · answers are based only on official scheme information.")

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.markdown("# 🩺 Ayushman Bharath Chatbot")
st.markdown(
    '<div class="subtitle">Ask questions about registration, eligibility, and benefits under Ayushman Bharath.</div>',
    unsafe_allow_html=True,
)


def handle_question(question_text: str, want_voice_reply: bool):
    st.session_state.messages.append({"role": "user", "content": question_text})
    recent_history = st.session_state.messages[-12:]

    with st.spinner(f"{question_text}..."):
        answer = get_answer(question_text, recent_history, st.session_state.language)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    if want_voice_reply:
        with st.spinner("Generating voice reply..."):
            st.session_state.reply_audio = synthesize(answer, st.session_state.language)
    else:
        st.session_state.reply_audio = None


# ---------------------------------------------------------------------------
# Voice input (English / Hindi only)
# ---------------------------------------------------------------------------
if voice_enabled:
    st.markdown('<div class="section-label">🎤 Speak your question</div>', unsafe_allow_html=True)
    audio_value = st.audio_input("Record your question", label_visibility="collapsed")
    if audio_value is not None:
        raw_bytes = audio_value.getvalue()
        if raw_bytes != st.session_state.last_audio_input_bytes:
            st.session_state.last_audio_input_bytes = raw_bytes
            with st.spinner("Transcribing..."):
                transcribed_text = transcribe(raw_bytes, st.session_state.language)
            if transcribed_text.strip():
                handle_question(transcribed_text, want_voice_reply=True)

    st.markdown('<div class="section-label">📁 Or upload a voice recording</div>', unsafe_allow_html=True)
    uploaded_audio = st.file_uploader(
        "Upload audio",
        type=["wav", "flac", "ogg", "opus", "mp3", "m4a", "aac"],
        label_visibility="collapsed",
    )
    if uploaded_audio is not None:
        raw_bytes = uploaded_audio.getvalue()
        if raw_bytes != st.session_state.last_uploaded_audio_bytes:
            st.session_state.last_uploaded_audio_bytes = raw_bytes
            with st.spinner("Transcribing uploaded audio..."):
                transcribed_text = transcribe(raw_bytes, st.session_state.language)
            if transcribed_text.strip():
                handle_question(transcribed_text, want_voice_reply=True)
            else:
                st.warning("Couldn't detect any speech in that file. Try a clearer recording.")

    st.divider()

# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------
if st.session_state.messages:
    st.markdown('<div class="section-label">💬 Conversation</div>', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🩺"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])
else:
    st.caption("No messages yet -- ask a question below to get started.")

# ---------------------------------------------------------------------------
# Text input -- always available; the only input method for Telugu/Nepali
# ---------------------------------------------------------------------------
user_input = st.chat_input("Type your question here...")
if user_input:
    handle_question(user_input, want_voice_reply=voice_enabled)
    st.rerun()

# --- Play the latest spoken reply, if any ---
if voice_enabled and st.session_state.reply_audio is not None:
    st.audio(st.session_state.reply_audio, format="audio/wav", autoplay=True)
