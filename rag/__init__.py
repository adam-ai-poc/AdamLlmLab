import sys
sys.path.append("..")
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .vectordb import *
from .load import *
from .chunk import *


ADAM_VECTORDB = {
    "chroma": ChromaDB
}

ADAM_LOADER = {
    "pdf": PDFLoader
}

ADAM_CHUNKER = {
    "recursiveCharacterTextSplitter": RecursiveChunker
}