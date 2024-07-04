import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from rag_app import load_chain

st.set_page_config(page_title="AdamLab: RAG", page_icon="app/images/A.png")
st.title("AdamLab: Rag App")

@st.cache_resource()
def initialize():
    load_dotenv()
    print("Initializing ragchain")
    ragchain = load_chain()
    return ragchain


# def get_stream_response(query, chat_history=None):
#     template = """
#     You are a helpful assistant. Answer the following questions considering the following:

#     Chat history: {chat_history}

#     User Question: {user_question}
#     """
#     prompt = ChatPromptTemplate.from_template(template)
#     llm = ChatOpenAI()
#     chain = prompt | llm | StrOutputParser()
#     return chain.stream({
#         "user_question": query,
#         "chat_history": chat_history
#     })

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
            # ai_response = st.write_stream(get_stream_response(user_query, st.session_state.chat_history))
            ai_response = st.write_stream(get_response(ragchain, user_query))
            # ai_response = st.markdown(get_response(ragchain, user_query))
        
        st.session_state.chat_history.append(AIMessage(ai_response)) 

if __name__ == "__main__":
    main()