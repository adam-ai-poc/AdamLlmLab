import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
# Graph
# Qdrant

EMBEDDING = {
    "openAI": OpenAIEmbeddings
}

'''
Base Vector Database class 
'''
class VectorDB:
    def __init__(self, vectorstore, embedding_model, persist_directory, debug=False):
        self.vectorstore=vectorstore
        self.vectorstore_name = self.vectorstore.__class__.__name__
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.debug = debug
        print(self.get_config_string()) if self.debug else None

    def store(self, chunks):
        self.num_chunks = self.num_chunks + len(chunks)
        self.vectorstore = self.vectorstore.from_documents(documents=chunks, embedding=self.embedding_model, persist_directory=self.persist_directory)

    def as_retriever(self, search_type, search_kwargs):
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        self.retriever = self.vectorstore.as_retriever(search_type=self.search_type, search_kwargs=self.search_kwargs)
        return self.retriever
    
    # Used when embedding model passed using kwargs
    def _init_embedding_model_from_dict(self, embedding_cfg:dict):
        backend = next(iter(embedding_cfg))
        backend_cfg = embedding_cfg[backend]
        return EMBEDDING[backend](**backend_cfg)
    
    def get_config_string(self):
        return f"""
            ==========================================
            Vector database configurations
            ------------------------------------------
            Vector Database: {self.vectorstore_name}
            Embedding model: {self.embedding_model.model}
            ==========================================
        """

    def get_chunk_count(self):
        return self.vectorstore._collection.count()
    
    def get_collection_name(self):
        return self.vectorstore.vectorstore._collection.name

'''
ChromaDB class 
'''
class ChromaDB(VectorDB):
    # Default
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_directory = None

    # If using kwargs, should pass in fields after the "chroma" key
    def __init__(self, embedding_model=embedding_model, persist_directory=persist_directory, debug=False, **kwargs):

        self.num_chunks = 0
        self.persist_directory = persist_directory

        # Parse embedding config
        if isinstance(embedding_model, dict):
            self.embedding_model = self._init_embedding_model_from_dict(embedding_model)
        else:
            self.embedding_model = embedding_model

        # VectorDB
        if os.path.exists(self.persist_directory):
            print(f"Loading from existing Chromadb located at: {self.persist_directory}")
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_model)
        else:
            print(f"No Chromadb located at: {self.persist_directory}, initializing new.")
            self.vectorstore = Chroma()

        self.debug=debug
        super().__init__(vectorstore=self.vectorstore, embedding_model=self.embedding_model, persist_directory=self.persist_directory, debug=self.debug)
