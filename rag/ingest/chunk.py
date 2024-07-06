from langchain_text_splitters import RecursiveCharacterTextSplitter

'''
Base Chunking class 
'''
class Chunker:
    def __init__(self, chunker, debug):
        self.chunker = chunker
        self.debug=debug
        print(self.get_config_string()) if self.debug else None

    def __call__(self, documents, vectorstore):
        self.documents = documents
        self.chunk_and_store(self.documents, vectorstore, self.debug)

    def chunk_and_store(self, documents, vectordb, debug=True):
        """Ingest data into the vector database."""
        self.chunks = self.chunker.split_documents(documents)
        if debug: 
            print("...Chunking...")
            print("Number of chunks: ", len(self.chunks))
            print("Chunks: ", self.chunks)
        # Vectordb should be passed by reference
        vectordb.store(chunks=self.chunks)
        print(f"{vectordb.num_chunks} Chunks saved into {vectordb.vectorstore_name} using embedding model: {vectordb.embedding_model.model}.") if debug else None
    
    def get_config_string(self):
        return NotImplementedError

'''
PDF Chunking class 
'''
class RecursiveChunker(Chunker):

    # Default
    chunk_size = 1000
    chunk_overlap = 0

    def __init__(self, chunk_size=chunk_size, chunk_overlap=chunk_overlap, debug=False, **kwargs):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        self.debug=debug
        super().__init__(chunker=self.chunker, debug=self.debug)

    def get_config_string(self):
        return f"""
            ==========================================
            Chunker configurations
            ------------------------------------------
            Chunker:  {self.chunker.__class__.__name__}
            Chunk Size: {self.chunk_size}
            Chunk Overlap: {self.chunk_overlap}
            ==========================================
            """
