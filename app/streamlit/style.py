import streamlit as st

def main_page_style():
    # Custom CSS to set background color
    st.markdown("""
        <style>
        button[kind="secondary"] {
            background-color: #abf7b1;
            padding: 20px;
            border-radius: 20px;
            border-width: 3px;
            cursor: pointer;
            text-align: center;  
            width: 300px;
            font-size: 16px;
        }
        button[kind="secondary"]:hover, button[kind="secondary"]:hover p, button[kind="secondary"]:active {
            background-color: lightblue;  /* Change background color on hover */
            border-color: #51A2D5;
            color: #082540;
        }
        button[kind="secondary"] p  {
            font-size: 20px;
            color: darkgreen;
        }

        </style>
        """, unsafe_allow_html=True)
    
# Hack to display title above page navigation 
def add_sidebar_title():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"]::before {
                content: "Pages";
                margin: 20px;
                font-size: 30px;
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def all_styles():
    main_page_style()
    add_sidebar_title()