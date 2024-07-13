import streamlit as st

def set_page_config(page_title):
    st.set_page_config(page_title=page_title, page_icon="🥼")

def set_title(title):
    st.title(title, anchor=False)