import getpass
import os
from langchain_openai import ChatOpenAI

def openai_llm():
    return ChatOpenAI(model="gpt-3.5-turbo-0125")