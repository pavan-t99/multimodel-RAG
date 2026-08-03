import streamlit as st
from Rag_pipeline import get_answer
from speech import transcribe, synthesize, warm_up_speech

st.set_page_config(page_title="Ayushman Bharath chatbot", layout="centered")

st.title("Ayushman Bharath chatbot")
st.write("Welcome to Ayushman Bharath")

LANGUAGES = ["English", "Hindi", "Telugu", "Nepali"]

# Languages with reliable, well-supported voice (ASR + TTS) quality.
# Telugu/Nepali text translation still works fine -- only voice is limited,
# so those two languages fall back to text-only chat below.
VOICE_SUPPORTED_LANGUAGES = {"English", "Hindi"}

if "language" not in st.session_state:
    st.session_state.language = "English"

# --- Sidebar: language selector lives here so it stays visible/pinned
# while the user scrolls through the conversation ---
with st.sidebar:
    st.header("Settings")
    st.session_state.language = st.selectbox(
        "Choose your language / भाषा चुनें / మీ భాషను ఎంచుకోండి / भाषा छान्नुहोस्",
        LANGUAGES,
        index=LANGUAGES.index(st.session_state.language),
    )

    if st.session_state.language not in VOICE_SUPPORTED_LANGUAGES:
        st.info(
            "Voice input/output for Telugu and Nepali is less accurate right now. "
            "For these languages, please type your question -- you'll still get "
            "answers in Telugu/Nepali text, just without voice."
        )

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


voice_enabled = st.session_state.language in VOICE_SUPPORTED_LANGUAGES

if voice_enabled:
    # --- Voice input: live mic (works when the browser/environment allows it) ---
    st.write("Speak your question:")
    audio_value = st.audio_input("Record your question", label_visibility="collapsed")
    if audio_value is not None:
        raw_bytes = audio_value.getvalue()
        if raw_bytes != st.session_state.last_audio_input_bytes:
            st.session_state.last_audio_input_bytes = raw_bytes
            with st.spinner("Transcribing..."):
                transcribed_text = transcribe(raw_bytes, st.session_state.language)
            if transcribed_text.strip():
                handle_question(transcribed_text, want_voice_reply=True)

    # --- Voice input: upload a recording (fallback when live mic isn't available,
    # e.g. Codespaces browser context blocking mic permission) ---
    st.write("Or upload a voice recording of your question:")
    uploaded_audio = st.file_uploader(
        "Upload audio", type=["wav", "flac", "ogg", "opus", "mp3", "m4a", "aac"],
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

# --- Text input (always available; the only input method for Telugu/Nepali) ---
user_input = st.chat_input("Or type your question")
if user_input:
    handle_question(user_input, want_voice_reply=voice_enabled)

# --- Conversation history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Play the latest spoken reply, if any ---
if voice_enabled and st.session_state.reply_audio is not None:
    st.audio(st.session_state.reply_audio, format="audio/wav", autoplay=True)
