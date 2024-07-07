import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, BASE_DIR)
from rag.ragchain import RagChain
from rag.utils import read_all_config, check_persist_directory
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/get_response', methods=["POST"])
def get_response():
    data = request.get_json()
    print(data)
    query = data["question"]
    response = ragchain(query)
    return jsonify({"response": response})



if __name__ == "__main__":
    RAG_CONFIG = read_all_config(os.path.join(BASE_DIR, "configs/rag.yaml"))

    # Temporary: If vectordb given by config file does not exists, create one and ingest into it.
    if not check_persist_directory(RAG_CONFIG):
        import subprocess
        ingestion_script_path = os.path.join(BASE_DIR, "scripts/ingest.py")
        subprocess.run(["python", ingestion_script_path], check=True, text=True)
        
    ragchain = RagChain(**RAG_CONFIG)
    app.run(debug=True)