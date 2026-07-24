import streamlit as st
from Rag_pipeline import get_answer

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

user_input = st.chat_input("Ask your question")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    recent_history = st.session_state.messages[-12:]
    with st.spinner(f"{user_input}..."):
        answer = get_answer(user_input, recent_history, st.session_state.language)
    st.session_state.messages.append({"role": "assistant", "content": answer})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])