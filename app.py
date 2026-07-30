import streamlit as st
from Rag_pipeline import get_answer
from speech import transcribe, synthesize, warm_up_speech

st.title("Ayushman Bharath chatbot")
st.write("Welcome to Ayushman Bharath")

LANGUAGES = ["English", "Hindi", "Telugu", "Nepali"]

if "language" not in st.session_state:
    st.session_state.language = "English"

st.session_state.language = st.selectbox(
    "Choose your language / भाषा चुनें / మీ భాషను ఎంచుకోండి / भाषा छान्नुहोस्",
    LANGUAGES,
    index=LANGUAGES.index(st.session_state.language),
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


def handle_question(question_text: str):
    st.session_state.messages.append({"role": "user", "content": question_text})
    recent_history = st.session_state.messages[-12:]

    with st.spinner(f"{question_text}..."):
        answer = get_answer(question_text, recent_history, st.session_state.language)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.spinner("Generating voice reply..."):
        st.session_state.reply_audio = synthesize(answer, st.session_state.language)


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
            handle_question(transcribed_text)

# --- Voice input: upload a recording (fallback when live mic isn't available,
# e.g. Codespaces browser context blocking mic permission) ---
st.write("Or upload a voice recording of your question:")
uploaded_audio = st.file_uploader(
    "Upload audio", type=["wav", "flac", "ogg","opus", "mp3", "m4a", "aac"], label_visibility="collapsed"
)
if uploaded_audio is not None:
    raw_bytes = uploaded_audio.getvalue()
    if raw_bytes != st.session_state.last_uploaded_audio_bytes:
        st.session_state.last_uploaded_audio_bytes = raw_bytes
        with st.spinner("Transcribing uploaded audio..."):
            transcribed_text = transcribe(raw_bytes, st.session_state.language)
        if transcribed_text.strip():
            handle_question(transcribed_text)
        else:
            st.warning("Couldn't detect any speech in that file. Try a clearer recording.")

# --- Text input (still available) ---
user_input = st.chat_input("Or type your question")
if user_input:
    handle_question(user_input)

# --- Conversation history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Play the latest spoken reply, if any ---
if st.session_state.reply_audio is not None:
    st.audio(st.session_state.reply_audio, format="audio/wav", autoplay=True)