import os
import sys
import warnings
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '../'))
warnings.filterwarnings("ignore")

import argparse
from rag.utils import read_all_config, benchmark
from rag.ragchain import RagChain
from model.prompt import zero_shot_prompt
from model.llm import OpenaiLLM

def parse_args():
    parser = argparse.ArgumentParser(description="A terminal based app to interact with rag app. Make sure ingestion is done before running this script.")

    # Add arguments
    parser.add_argument('-g', "--config", type=str, default=os.path.join(SCRIPT_DIR, "../configs/rag.yaml"), help="Rag config path.")
    parser.add_argument('-d', "--debug", action='store_true', help="Enable debug mode.")

    return vars(parser.parse_args())

@benchmark
def load_chain(config, debug):
    print(config, debug)
    RAG_CONFIG = read_all_config(config)
    print(RAG_CONFIG)
    ragchain = RagChain(**RAG_CONFIG, debug=debug)

    return ragchain

def main():
    args = parse_args()
    # Initialize chain
    ragchain = load_chain(**args)

    # Chat interface
    print("==========================================")
    print("Welcome to AdamLab's rag interface! Chat with your RagChain now! Type 'exit() to exit.")
    print("==========================================")
    print(f"LLM: {ragchain('Hi')}")
    while True:
        query = input("You: ")
        if query.strip().lower() == "exit()":
            print("Exiting the chat. Goodbye!")
            break
        else:
            # Echo the user input for now. You can add more logic here.
            print(f"LLM: {ragchain(query)}")


if __name__ == "__main__":
    main()