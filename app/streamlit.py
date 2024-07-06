import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '../'))

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from rag.utils import read_all_config
from rag.ragchain import RagChain

st.set_page_config(page_title="AdamLab: RAG", page_icon="app/images/A.png")
st.title("AdamLab: Rag App")

@st.cache_resource()
def initialize():
    load_dotenv()
    config_file = os.path.join(SCRIPT_DIR, "../configs/rag.yaml")
    print("Initializing ragchain")
    RAG_CONFIG = read_all_config(config_file)
    ragchain = RagChain(**RAG_CONFIG)

    return ragchain

# Response
def get_response(ragchain, query):
    return ragchain.stream(query)

def main():
    ragchain = initialize()
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
            ai_response = st.write_stream(get_response(ragchain, user_query))
        
        st.session_state.chat_history.append(AIMessage(ai_response)) 

if __name__ == "__main__":
    main()