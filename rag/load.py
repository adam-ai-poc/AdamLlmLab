import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from .utils import read_config
from typing import Union

LOADER_CONFIG = read_config(os.path.join(os.path.dirname(__file__), "config.yaml"), "ragchain")["ingestion_config"]["loader"]

class AdamPyPDFLoader(PyPDFLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

LOADER = {
    "pdf": AdamPyPDFLoader
}

class Loader:
    def __init__(self, loader, loader_cfg:dict, debug):
        self.loader = loader
        self.loader_cfg = loader_cfg
        self.documents = []
        self.doc_paths = []
        self.debug = debug
        if self.debug:
            print("Loader: ", loader.__name__)

    def __call__(self, doc_path:Union[str, list]):
        return self.load(doc_path, self.debug)
    
    def load(self, doc_path:Union[str, list], debug=False):

        if isinstance(doc_path, list):
            for path in doc_path:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File {path} not found")
                else:
                    loader_config = self.loader_cfg
                    loader_config["file_path"] = path
                    print(self.loader_cfg) if debug else None
                    self.loader_instance = self.loader(**loader_config)
                    self.documents.extend(self.loader_instance.load())
                    self.doc_paths.append(path)
        else:
            if not os.path.exists(doc_path):
                raise FileNotFoundError(f"File {doc_path} not found")
            else:
                loader_config = self.loader_cfg
                loader_config["file_path"] = doc_path
                print(self.loader_cfg) if debug else None
                self.loader_instance = self.loader(**loader_config)
                self.documents.extend(self.loader_instance.load())
                self.doc_paths.append(doc_path)
        return self.documents

class PDFLoader(Loader):
    loader_cfg = LOADER_CONFIG["pdf"]

    def __init__(self, loader_cfg:dict=loader_cfg, debug=False):
        self.loader_type = "pdf"
        self.loader_cfg = loader_cfg[self.loader_type]
        self.loader = LOADER[self.loader_type]
        self.debug = debug

        super().__init__(loader=self.loader, loader_cfg=self.loader_cfg, debug=debug)

