import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from .utils import read_config
# Graph
# Qdrant

VECTORDB_CONFIG = read_config(os.path.join(os.path.dirname(__file__), "config.yaml"), "vectordb_config")
EMBEDDING = {
    "openAI": OpenAIEmbeddings
}

class VectorDB:
    def __init__(self, vectordb, embedding_model, debug):
        self.vectordb=vectordb
        self.vectordb_name = self.vectordb.__class__.__name__
        self.embedding_model = embedding_model
        self.debug = debug
        if self.debug:
            print("==========================================")
            print("Vector database configurations")
            print("==========================================")
            print("Vectordb:", self.vectordb_name)
            print("Embedding model:", embedding_model.model)
            print("==========================================")

    def store(self, chunks):
        self.vectordb = self.vectordb.from_documents(chunks, self.embedding_model)


class ChromaDB(VectorDB):
    # Default
    vectordb_cfg = VECTORDB_CONFIG["chroma"]

    def __init__(self, vectordb_cfg: dict=vectordb_cfg, debug=False):
        # VectorDB
        self.vectordb = Chroma()
        self.num_chunks = 0

        # Embedding
        self.embedding_backend = next(iter(self.vectordb_cfg.get("embedding"))) 
        self.embedding_model_name = self.vectordb_cfg["embedding"][self.embedding_backend]["model_name"]
        self.embedding_model = EMBEDDING[self.embedding_backend](model=self.embedding_model_name)

        self.debug=debug
        super().__init__(vectordb=self.vectordb, embedding_model=self.embedding_model, debug=self.debug)