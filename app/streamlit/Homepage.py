import streamlit as st
from style import all_styles
from st_click_detector import click_detector
from streamlit_extras.switch_page_button import switch_page

def set_page_config():
    st.set_page_config(page_title="AdamLab", page_icon="🥼")

def set_title():
    st.title("Main Page", anchor=False)

def set_sidebar():
    st.sidebar.markdown("Pages")
    st.sidebar.success("Select the pages above.")

def option_ingest():
    st.write("Load, Chunk, and embed your documents into a vector database :")
    if st.button("Ingestion"):
        switch_page("Ingestion")

def option_chat():
    st.write("Interact with a Large Language Model  \n (Openai, HuggingFace) :")
    if st.button("Chat Playground"):
        switch_page("Chat")

def display_options():
    col1, col2 = st.columns(2, vertical_alignment="top")
    with col1:
        option_ingest()
    with col2:
        option_chat()



set_page_config()
all_styles()
set_sidebar()
set_title()
display_options()
