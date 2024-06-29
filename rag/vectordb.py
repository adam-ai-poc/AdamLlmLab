import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from .utils import read_config
# Graph
# Qdrant

VECTORDB_CONFIG = read_config(os.path.join(os.path.dirname(__file__), "config.yaml"), "ragchain")["vectordb_config"]

EMBEDDING = {
    "openAI": OpenAIEmbeddings
}

'''
Base Vector Database class 
'''
class VectorDB:
    def __init__(self, vectorstore, embedding_model, debug):
        self.vectorstore=vectorstore
        self.vectorstore_name = self.vectorstore.__class__.__name__
        self.embedding_model = embedding_model
        self.debug = debug
        if self.debug:
            print("==========================================")
            print("Vector database configurations")
            print("==========================================")
            print("Vector Database:", self.vectorstore_name)
            print("Embedding model:", embedding_model.model)
            print("==========================================")

    def store(self, chunks):
        self.vectorstore = self.vectorstore.from_documents(chunks, self.embedding_model)

    def as_retriever(self, search_type, search_kwargs):
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        self.retriever = self.vectordb.as_retriever(search_type=self.search_type, search_kwargs=self.search_kwargs)
        return self.retriever

'''
ChromaDB class 
'''
class ChromaDB(VectorDB):
    # Default
    vectordb_cfg = VECTORDB_CONFIG["chroma"]

    def __init__(self, vectordb_cfg: dict=vectordb_cfg, debug=False):
        # VectorDB
        self.vectorstore = Chroma()
        self.num_chunks = 0

        # Embedding
        self.embedding_backend = next(iter(self.vectordb_cfg.get("embedding"))) 
        self.embedding_model_name = self.vectordb_cfg["embedding"][self.embedding_backend]["model_name"]
        self.embedding_model = EMBEDDING[self.embedding_backend](model=self.embedding_model_name)

        self.debug=debug
        super().__init__(vectorstore=self.vectorstore, embedding_model=self.embedding_model, debug=self.debug)

    def chunk_count(self):
        return self.vectorstore._collection.count()