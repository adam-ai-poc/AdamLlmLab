from . import ADAM_VECTORDB, ADAM_CHUNKER, ADAM_LOADER
from ..utils import read_config
from typing import Union

class AdamIngest:
    def __init__(self, vectordb, loader, chunker, debug=False, **kwargs):

        self.debug = debug

        # Vector db initialization
        if isinstance(vectordb, dict):
            self.vectordb = self._init_component_from_dict(ADAM_VECTORDB, vectordb)
        else:
            self.vectordb = vectordb
        print(f"Vectordb: {self.vectordb} Initialized.")

        # Loader initialization
        if isinstance(loader, dict):
            self.loader = self._init_component_from_dict(ADAM_LOADER, loader)
        else:
            self.loader = loader
        print(f"Loader: {self.loader} Initialized.")

        # Chunker initialization
        if isinstance(chunker, dict):
            self.chunker = self._init_component_from_dict(ADAM_CHUNKER, chunker)
        else:
            self.chunker = chunker
        print(f"Chunker: {self.chunker} Initialized.")

    def _init_component_from_dict(self, COMPONENT, component_cfg:dict):
        component = next(iter(component_cfg))
        component_parameters_cfg = component_cfg[component]
        return COMPONENT[component](**component_parameters_cfg, debug = self.debug)


    def ingest(self, doc_path:Union[str, list]):
        documents = self.loader.load(doc_path)
        self.chunker.chunk_and_store(documents, self.vectordb)