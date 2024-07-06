import os
import sys
import warnings
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '../'))
warnings.filterwarnings("ignore")

import argparse
from rag.utils import benchmark, read_all_config
from rag.ingest.ingest import *

def parse_args():
    parser = argparse.ArgumentParser(description="A script to ingest your documents.")

    # Add arguments
    parser.add_argument('-p', "--document_path", type=str, default=os.path.join(SCRIPT_DIR, "../docs/TayXueHao-Resume.pdf"), help="Document path or directory.")
    parser.add_argument('-g', "--config", type=str, default=os.path.join(SCRIPT_DIR, "../configs/ingest.yaml"), help="Ingest config path.")
    parser.add_argument('-d', "--debug", action='store_true', help="Enable debug mode.")

    return vars(parser.parse_args())

@benchmark
def ingest_documents(document_path, config, debug):
    INGESTION_CONFIG = read_all_config(config)
    file_type = next(iter(INGESTION_CONFIG["loader"]))
    
    ingestor = AdamIngest(**INGESTION_CONFIG, debug=debug)
    if os.path.isfile(document_path):
        document_path = document_path
    elif os.path.isdir(document_path):
        document_path = [os.path.join(document_path, f) for f in os.listdir(document_path) 
                      if os.path.isfile(os.path.join(document_path, f)) and f.lower().endswith(f".{file_type}")]
    else:
        document_path = None
    ingestor.ingest(document_path)


def main():
    # Ingest
    args = parse_args()
    ingest_documents(**args)
    print("Documents ingested")

if __name__ == "__main__":
    main()