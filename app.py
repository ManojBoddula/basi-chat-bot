from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()  # works locally, ignored on Streamlit Cloud

model = ChatOpenAI(
    model="allenai/olmo-3.1-32b-think:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=False
    )

memory = st.session_state.memory

prompt_template = PromptTemplate(
    input_variables=["chat_history", "input"],
    template="{chat_history}\nHuman: {input}\nAI:"
)

conversation_runnable = prompt_template | model

def system_prompts(user_input: str):
    text = user_input.lower()

    if text in ["hi", "hello", "hey"]:
        return "Hello! How can I help you?"

    if "bye" in text:
        return "Goodbye! Have a nice day 😊"

    chat_history = memory.load_memory_variables({})["chat_history"]

    ai_response = conversation_runnable.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    ai_text = ai_response.content
    memory.save_context({"input": user_input}, {"output": ai_text})

    return ai_text

st.title("Chat-Bot")
st.write("Welcome to my first basic chat bot")

if "chat_history_ui" not in st.session_state:
    st.session_state.chat_history_ui = []

user_input = st.text_input("Your Question")

if st.button("Ask AI") and user_input.strip():
    with st.spinner("Thinking..."):
        response = system_prompts(user_input)

    st.session_state.chat_history_ui.append((user_input, response))

    st.subheader("AI Answer")
    st.write(response)
