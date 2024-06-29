import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from .utils import read_config

CHUNKER_CONFIG = read_config(os.path.join(os.path.dirname(__file__), "config.yaml"), "ragchain")["ingestion_config"]["chunker"]

CHUNKER = {
    "recursiveCharacterTextSplitter": RecursiveCharacterTextSplitter
}

'''
Base Ingestion class 
'''
class Chunker:
    def __init__(self, chunker, debug):
        self.chunker = chunker
        self.debug=debug
        if self.debug:
            print("chunker:", chunker.__class__.__name__)

    def __call__(self, documents, db):
        self.documents = documents
        self.ingest(self.documents, db, self.debug)

    def ingest(self, documents, db, debug=True):
        """Ingest data into the vector database."""
        print("Documents:", documents) if debug else None

        self.chunks = self.chunker.split_documents(documents)
        print("Number of chunks: ", len(self.chunks)) if debug else None
        print("Chunks: ", self.chunks) if debug else None
        # Vectordb should be passed by reference
        db.store(chunks=self.chunks)
        db.num_chunks = db.num_chunks + len(self.chunks)
        print(f"{db.num_chunks} Chunks saved into {db.vectordb_name} using embedding model: {db.embedding_model}.")

'''
PDF Ingestion class 
'''
class RecursiveChunker(Chunker):

    # Default
    chunker_cfg = CHUNKER_CONFIG

    def __init__(self, chunker_cfg: dict=chunker_cfg, debug=False):
        self.chunker_cfg = chunker_cfg
        self.chunking_technique = next(iter(self.chunker_cfg))
        self.chunker = CHUNKER[self.chunking_technique]
        self.chunker = self.chunker(**self.chunker_cfg[self.chunking_technique]) # **kwargs

        self.debug=debug

        super().__init__(chunker=self.chunker, debug=self.debug)
