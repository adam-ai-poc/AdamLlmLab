from . import *
from .vectordb import *
from .load import *
from .chunk import *
from .retrieve import *
from model.llm import *
from model.prompt import *

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

RAGCHAIN_CONFIG = read_config(os.path.join(os.path.dirname(__file__), "config.yaml"), "ragchain")

'''
Chain class to link the rag pipelines
'''
class RagChain:

    ragchain_cfg = RAGCHAIN_CONFIG

    def __init__(self, vectordb=None, loader=None, chunker=None, retriever=None, system_prompt=None, llm=None, ragchain_cfg:dict={}, debug=False):
        self.debug = debug
        self.system_prompt = system_prompt
        self.llm = llm
        if ragchain_cfg:
            print("...Ragchain config found. Creating or recreating rag piplines...") if debug else None
            self.create_pipelines(ragchain_cfg=ragchain_cfg, debug=debug)
        else:
            self.vectordb = vectordb
            self.loader = loader
            self.chunker = chunker
            self.rertiever = retriever
            self.system_prompt = system_prompt
            self.llm = llm

    def create_pipelines(self, ragchain_cfg:dict=ragchain_cfg, debug=False):
        self.vectordb_cfg = ragchain_cfg["vectordb_config"]
        self.loader_cfg = ragchain_cfg["ingestion_config"]["loader"]
        self.chunker_cfg = ragchain_cfg["ingestion_config"]["chunker"]
        self.retrieval_cfg = ragchain_cfg["retrieval_config"]

        if debug:
            print("==========================================")
            print("Ragchain configurations")
            print("==========================================")
            print("Vector database: ", self.vectordb_cfg)
            print("Loader         : ", self.loader_cfg)
            print("Chunker        : ", self.chunker_cfg)
            print("Retrieval      : ", self.retrieval_cfg[next(iter(self.retrieval_cfg))])
            print("Prompt         : ", self.system_prompt)
            print("LLM            : ", self.llm.model_name)
            print("==========================================")

        self.vectordb = ADAM_VECTORDB[next(iter(self.vectordb_cfg))](self.vectordb_cfg, debug=debug)
        self.loader = ADAM_LOADER[next(iter(self.loader_cfg))](self.loader_cfg, debug=debug)
        self.chunker = ADAM_CHUNKER[next(iter(self.chunker_cfg))](self.chunker_cfg, debug=debug)

    def chain(self, doc_path:Union[str, list], retriever=None, debug=False):
        self.documents = self.loader.load(doc_path=doc_path, debug=debug)
        self.chunks = self.chunker.chunk(documents=self.documents, vectordb=self.vectordb, debug=debug)
        
        # if next(iter(self.retrieval_cfg)) == "vectorstore":
        if retriever:
            self.retriever = retriever
        else:
            self.retriever = VectorStoreRetriever(self.vectordb, retrieval_cfg=self.retrieval_cfg[next(iter(self.retrieval_cfg))], debug=debug)

        if debug:
            import langchain
            langchain.debug = True

        self.ragchain = {"context": self.retriever.retriever | self.format_docs, "input": RunnablePassthrough()} \
            | self.system_prompt \
            | self.llm \
            | StrOutputParser()
        
        return self.ragchain

    def clear_vectorstore():
        pass

    # Preprocessing
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)