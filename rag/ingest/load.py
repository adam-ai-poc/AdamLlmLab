import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from typing import Union

class AdamPyPDFLoader(PyPDFLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Loader:
    def __init__(self, loader, loader_cfg, debug):
        self.loader = loader
        self.loader_cfg = loader_cfg
        self.documents = []
        self.doc_paths = []
        self.debug = debug
        print(self.get_config_string()) if self.debug else None

    def __call__(self, doc_path:Union[str, list]):
        return self.load(doc_path, self.debug)
    
    def load(self, doc_path:Union[str, list], debug=False):

        if isinstance(doc_path, list):
            for path in doc_path:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File {path} not found")
                else:
                    self._load_doc_from_path(path)
        else:
            if not os.path.exists(doc_path):
                raise FileNotFoundError(f"File {doc_path} not found")
            else:
                self._load_doc_from_path(doc_path)
        return self.documents
    
    def _load_doc_from_path(self, doc_path):
        # Use dictionary for loader arguments in case different loaders have different parameters
        self.loader_instance = self.loader(file_path=doc_path, **self.loader_cfg)
        self.documents.extend(self.loader_instance.load())
        self.doc_paths.append(doc_path)

    
    def get_config_string(self):
        return f"""
            ==========================================
            Loader configurations
            ------------------------------------------
            Loader:  {self.loader.__name__}
            Extract images: {self.loader_cfg}
            ==========================================
            """

class PDFLoader(Loader):
    extract_images = False

    def __init__(self, extract_images=extract_images, debug=False, **kwargs):
        self.extract_images = extract_images
        self.loader = AdamPyPDFLoader

        self.loader_cfg = {}
        self.loader_cfg["extract_images"] = extract_images

        self.debug = debug
        super().__init__(loader=self.loader, loader_cfg=self.loader_cfg, debug=debug)

