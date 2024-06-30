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
    def __init__(self, vectorstore, embedding_model, vectordb_cfg:dict={}, debug=False):
        self.vectorstore=vectorstore
        self.vectorstore_name = self.vectorstore.__class__.__name__
        self.embedding_model = embedding_model
        self.vectordb_cfg = vectordb_cfg
        self.debug = debug
        if self.debug:
            print("==========================================")
            print("Vector database configurations")
            print("==========================================")
            print("Vector Database:", self.vectorstore_name)
            print("Embedding model:", self.embedding_model.model)
            print("Vector Database configs: ", self.vectordb_cfg)
            print("==========================================")

    def store(self, chunks):
        self.vectordb_cfg["documents"] = chunks
        self.vectorstore = self.vectorstore.from_documents(**self.vectordb_cfg)

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
        self.vectordb_cfg = vectordb_cfg["chroma"]
        self.num_chunks = 0

        # Parse embedding config
        self.embedding_backend = next(iter(self.vectordb_cfg.get("embedding"))) 
        self.embedding_cfg = self.vectordb_cfg["embedding"][self.embedding_backend]
        self.embedding_model = EMBEDDING[self.embedding_backend](**self.embedding_cfg)

        # Setup kwargs for from_documents
        self.vectordb_cfg["embedding"] = self.embedding_model

        self.debug=debug
        super().__init__(vectorstore=self.vectorstore, embedding_model=self.embedding_model, vectordb_cfg=self.vectordb_cfg, debug=self.debug)

    def chunk_count(self):
        return self.vectorstore._collection.count()