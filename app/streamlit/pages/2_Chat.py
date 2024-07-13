import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..")
sys.path.insert(0, BASE_DIR)

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from rag.utils import read_all_config, check_persist_directory
from rag.ragchain import RagChain

from utils import set_page_config, set_title
from style import add_sidebar_title

@st.cache_resource()
def initialize():
    load_dotenv()
    config_file = os.path.join(BASE_DIR, "configs", "rag.yaml")
    print("Initializing ragchain")
    RAG_CONFIG = read_all_config(config_file)
    create_vectordb(RAG_CONFIG)
    ragchain = RagChain(**RAG_CONFIG) # debug = True

    return ragchain

# Temporary: If vectordb given by config file does not exists, create one and ingest into it.
def create_vectordb(RAG_CONFIG):
    if not check_persist_directory(RAG_CONFIG):
        import subprocess
        ingestion_script_path = os.path.join(BASE_DIR, "scripts/ingest.py")
        subprocess.run(["python", ingestion_script_path], check=True, text=True)

# Response
def get_response(query):
    return st.session_state.ragchain.stream(query)

def set_llm_configurations():
    st.session_state.ragchain.llm.temperature = st.session_state.temperature
    st.session_state.ragchain.llm.max_tokens = st.session_state.max_tokens
    st.session_state.ragchain.num_history = st.session_state.num_history
    print("Session configurations: ", {
        "temperature": st.session_state.temperature,
        "max_tokens": st.session_state.max_tokens,
        "num_history": st.session_state.num_history,
    })

def chat():
    # Debugging:
    # st.write(ragchain.llm.temperature)

    st.session_state.ragchain = initialize()
    set_llm_configurations()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("AdamBot"):
                st.markdown(message.content)

    user_query = st.chat_input("Your message")

    if user_query is not None and user_query !="":
        # Store conversation history in the frontend session
        st.session_state.chat_history.append(HumanMessage(user_query)) 

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AdamBot"):
            # Streaming
            ai_response = st.write_stream(get_response(user_query))
        
        st.session_state.chat_history.append(AIMessage(ai_response)) 

def llm_configurations():
    st.sidebar.title("Chat Configurations:")
    st.session_state.num_history = st.sidebar.slider("Number of Chat History", 0, 5, 3)
    st.session_state.temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0)
    st.session_state.max_tokens = st.sidebar.slider("Max Tokens", 0, 2048, 256)

set_page_config("AdamLab: Chat Playground")
set_title("Chat Playground")
add_sidebar_title()

llm_configurations()
chat()

