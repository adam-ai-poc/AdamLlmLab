import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .utils import read_config
from typing import Any, Optional, List

INGEST_CONFIG = read_config(os.path.join(os.path.dirname(__file__), "config.yaml"), "ingestion_config")
VECTORDB_CONFIG = read_config(os.path.join(os.path.dirname(__file__), "config.yaml"), "vectordb_config")
VECTORDB = {
    "chroma": Chroma
}
SPLITTER = {
    "recursiveCharacterTextSplitter": RecursiveCharacterTextSplitter
}
EMBEDDING = {
    "openAI": OpenAIEmbeddings
}

'''
Base Ingestion class 
'''
class Ingestor:
    def __init__(self, doc_path, vectordb, loader, splitter, embedding_model):
        self.vectordb = vectordb
        self.loader = loader
        self.splitter = splitter
        self.embedding_model = embedding_model
        self.doc_path = doc_path

    def __call__(self):
        docs = self.loader.load()
        print(docs)
        chunks = self.splitter.split_documents(docs)
        vectorstore = self.vectordb.from_documents(documents=chunks, embedding=self.embedding_model)
        return vectorstore
    
'''
PDF Ingestion service 
'''
class PdfIngestor(Ingestor):

    ingestion_cfg = INGEST_CONFIG
    vectordb_cfg = VECTORDB_CONFIG

    def __init__(self, ingestion_cfg: dict=ingestion_cfg, vectordb_cfg: dict=vectordb_cfg):
        # Document
        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.doc_path = os.path.join(self.current_directory, "../docs/TayXueHao-Resume.pdf")
        # VectorDB
        self.pdf_config = ingestion_cfg["pdf"]
        self.vectordb_name = vectordb_cfg["vectordb"]
        self.vectordb = VECTORDB[self.vectordb_name]
        # Chunker
        self.loader = PyPDFLoader(file_path=self.doc_path)
        self.splitter_config = self.pdf_config["splitter"]
        self.splitting_technique = next(iter(self.splitter_config))
        self.splitter = SPLITTER[self.splitting_technique]
        self.splitter = self.splitter(**self.splitter_config[self.splitting_technique]) # **kwargs
        # Embedding
        self.embedding_backend = next(iter(self.pdf_config.get("embedding"))) 
        self.embedding_model_name = self.pdf_config["embedding"][self.embedding_backend]["model_name"]
        self.embedding_model = EMBEDDING[self.embedding_backend](model=self.embedding_model_name)
        super().__init__(doc_path=self.doc_path, vectordb=self.vectordb, loader=self.loader, splitter=self.splitter, embedding_model=self.embedding_model)

