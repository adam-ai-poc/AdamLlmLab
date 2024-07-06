from .ingest import ADAM_VECTORDB
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

'''
Base Retriever class 
'''
class Retriever:
    def __init__(self):
        pass

    def __call__(self):
        pass

    def get_config_string(self):
        pass

    def _init_component_from_dict(self, ADAM_COMPONENT, component_cfg:dict):
        component = next(iter(component_cfg))
        component_parameters_cfg = component_cfg[component]
        return ADAM_COMPONENT[component](**component_parameters_cfg, debug = self.debug)

'''
Vector Store Retriever class 
'''
class VectorStoreRetriever(Retriever):

    def __init__(self, vectordb, search_type, search_kwargs, debug=False, **kwargs):

        self.debug = debug
        # Vector db initialization
        if isinstance(vectordb, dict):
            self.vectordb = self._init_component_from_dict(ADAM_VECTORDB, vectordb)
        else:
            self.vectordb = vectordb
        print(f"Vectordb: {self.vectordb} Initialized.")

        self.search_type = search_type
        self.search_kwargs = search_kwargs

        print(self.get_config_string()) if self.debug else None
        self.set_retriever_params(search_type=self.search_type, search_kwargs=self.search_kwargs)
        print(f"Vector Store Retriever: {self.retriever} created")

    # Can be called to change parameters without reinitializing vectordbs
    def set_retriever_params(self, search_type, search_kwargs, **kwargs):
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        self.retriever = self.vectordb.vectorstore.as_retriever(search_type=self.search_type, search_kwargs=self.search_kwargs)

    def __call__(self, query):
        return self.retrieve(query, debug=self.debug)

    def retrieve(self, query, debug=False):
        contexts = self.retriever.invoke(query)
        print("Contexts: ", contexts) if debug else None
        return contexts
    
    def get_config_string(self):
        return f"""
            ==========================================
            Retrieval configurations
            ------------------------------------------
            Retrieval type: {__class__.__name__}
            Search type: {self.search_type}
            Search config: {self.search_kwargs}
            ==========================================]
        """