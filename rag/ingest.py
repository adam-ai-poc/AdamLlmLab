import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .utils import read_config
from typing import Any, Optional, List

INGEST_CONFIG = read_config(os.path.join(os.path.dirname(__file__), "config.yaml"), "ingestion_config")

SPLITTER = {
    "recursiveCharacterTextSplitter": RecursiveCharacterTextSplitter
}

'''
Base Ingestion class 
'''
class Ingestor:
    def __init__(self, doc_path, loader, splitter, debug):
        self.loader = loader
        self.splitter = splitter
        self.doc_path = doc_path
        self.debug=debug
        if self.debug:
            print("==========================================")
            print("Ingestion configurations")
            print("==========================================")
            print("Loader:", loader.__class__.__name__)
            print("Splitter:", splitter.__class__.__name__)
            print("Document:", doc_path)
            print("==========================================")

    def __call__(self, db):
        self.ingest(db)

    def ingest(self, db):
        """Ingest data into the vector database."""
        self.docs = self.loader.load()
        print("Documents:", self.docs) if self.debug else None

        self.chunks = self.splitter.split_documents(self.docs)
        print("Number of chunks: ", len(self.chunks)) if self.debug else None
        print("Chunks: ", self.chunks) if self.debug else None
        # Vectordb should be passed by reference
        db.store(chunks=self.chunks)
        db.num_chunks = len(self.chunks)
        print(f"{db.num_chunks} Chunks saved into {db.vectordb_name} using embedding model: {db.embedding_model}.")

'''
PDF Ingestion class 
'''
class PdfIngestor(Ingestor):

    # Default
    ingestion_cfg = INGEST_CONFIG["pdf"]

    def __init__(self, doc_path, ingestion_cfg: dict=ingestion_cfg, debug=False):
        self.ingestion_cfg = ingestion_cfg
        # Document - If we want to ingest the same document to different vectordb for experiments, we only need to initialize doc once.
        self.doc_path = doc_path
        # Chunker
        self.loader = PyPDFLoader(file_path=self.doc_path)
        self.splitter_config = self.ingestion_cfg["splitter"]
        self.splitting_technique = next(iter(self.splitter_config))
        self.splitter = SPLITTER[self.splitting_technique]
        self.splitter = self.splitter(**self.splitter_config[self.splitting_technique]) # **kwargs

        self.debug=debug

        super().__init__(doc_path=self.doc_path, loader=self.loader, splitter=self.splitter, debug=self.debug)

