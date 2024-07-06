from . import ADAM_VECTORDB, ADAM_CHUNKER, ADAM_LOADER
from ..utils import read_config
from typing import Union

class AdamIngest:
    def __init__(self, vectordb, loader, chunker, debug=False, **kwargs):

        self.debug = debug

        # Vector db initialization
        self.vectordb = self.set_component(component=vectordb, ADAM_COMPONENT=ADAM_VECTORDB)
        print(f"Vectordb: {self.vectordb} Initialized.")

        # Loader initialization
        self.loader = self.set_component(component=loader, ADAM_COMPONENT=ADAM_LOADER)
        print(f"Loader: {self.loader} Initialized.")

        # Chunker initialization
        self.chunker = self.set_component(component=chunker, ADAM_COMPONENT=ADAM_CHUNKER)
        print(f"Chunker: {self.chunker} Initialized.")

    def set_component(self, component, ADAM_COMPONENT=None):
        if isinstance(component, dict):
            return self._init_component_from_dict(ADAM_COMPONENT, component)
        else:
            return component


    def _init_component_from_dict(self, ADAM_COMPONENT, component_cfg:dict):
        component = next(iter(component_cfg))
        component_parameters_cfg = component_cfg[component]
        return ADAM_COMPONENT[component](**component_parameters_cfg, debug = self.debug)


    def ingest(self, doc_path:Union[str, list]):
        documents = self.loader.load(doc_path)
        self.chunker.chunk_and_store(documents, self.vectordb)