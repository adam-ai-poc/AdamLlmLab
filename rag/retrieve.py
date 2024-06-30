import os

from.utils import read_config
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
# Unsupported
# from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
# from langchain.retrievers.merger_retriever import MergerRetriever
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain.retrievers.re_phraser import RePhraseQueryRetriever
# from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain.retrievers.time_weighted_retriever import (
#     TimeWeightedVectorStoreRetriever,
# )

RETRIEVAL_CONFIG = read_config(os.path.join(os.path.dirname(__file__), "config.yaml"), "ragchain")["retrieval_config"]

'''
Base Retriever class 
'''
class Retriever:
    def __init__(self):
        pass

    def __call__(self):
        pass

'''
Vector Store Retriever class 
'''
class VectorStoreRetriever(Retriever):

    retrieval_cfg = RETRIEVAL_CONFIG["vectorstore"]

    def __init__(self, vectordb, retrieval_cfg: dict=retrieval_cfg, debug=False):
        self.vectordb = vectordb
        self.retrieval_cfg = retrieval_cfg
        self.search_type = self.retrieval_cfg["search_type"]
        self.search_cfg = self.retrieval_cfg["search_config"]
        self.debug = debug
        if self.debug:
            print("==========================================")
            print("Retrieval configurations")
            print("==========================================")
            print("Retrieval type: ", __class__.__name__)
            print("Search type: ", self.search_type)
            print("Search config: ", self.search_cfg)
            print("==========================================")
        self.retriever = self.vectordb.vectorstore.as_retriever(search_type=self.search_type, search_kwargs=self.search_cfg)

    def __call__(self, query):
        return self.retrieve(query, debug=self.debug)

    def retrieve(self, query, debug=False):
        contexts = self.retriever.invoke(query)
        print("Contexts: ", contexts) if debug else None
        return contexts