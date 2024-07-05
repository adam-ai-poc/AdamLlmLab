from .vectordb import ChromaDB
from .load import PDFLoader
from .chunk import RecursiveChunker

ADAM_VECTORDB = {
    "chroma": ChromaDB
}

ADAM_LOADER = {
    "pdf": PDFLoader
}

ADAM_CHUNKER = {
    "recursiveCharacterTextSplitter": RecursiveChunker
}
