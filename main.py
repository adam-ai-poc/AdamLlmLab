import argparse
import os
import warnings
warnings.filterwarnings("ignore")

from rag.utils import read_config, benchmark
from rag.ragchain import RagChain
from model.prompt import zero_shot_prompt
from model.llm import OpenaiLLM

def parse_args():
    parser = argparse.ArgumentParser(description="A simple argument parser example")

    # Add arguments
    parser.add_argument('-p', "--document_path", type=str, default="docs/TayXueHao-Resume.pdf", help="Document path or directory.")
    parser.add_argument('-g', "--config", type=str, default="configs/rag.yaml", help="Rag config path.")
    parser.add_argument('-d', "--debug", action='store_true', help="Enable debug mode.")

    return parser.parse_args()

@benchmark
def load_chain(args):
    args = parse_args()

    RAG_CONFIG = read_config(args.config, "ragchain")
    file_type = next(iter(RAG_CONFIG["ingestion_config"]["loader"]))

    prompt_template = zero_shot_prompt()
    llm = OpenaiLLM()
    chain = RagChain(system_prompt=prompt_template, llm=llm, ragchain_cfg=RAG_CONFIG, debug=args.debug)
    if os.path.isfile(args.document_path):
        document_path = args.document_path
    elif os.path.isdir(args.document_path):
        document_path = [os.path.join(args.document_path, f) for f in os.listdir(args.document_path) 
                      if os.path.isfile(os.path.join(args.document_path, f)) and f.lower().endswith(f".{file_type}")]
    else:
        document_path = None
    if args.debug:
        print("==========================================")
        print("Documents to load: ", document_path)
        print("==========================================")
    ragchain = chain.chain(doc_path=document_path, debug=args.debug)

    return ragchain

def main():
    args = parse_args()
    # Initialize chain
    ragchain = load_chain(args)

    # Chat interface
    print("==========================================")
    print("Welcome to AdamLLM's rag interface! Chat with your RagChain now! Type 'quit() to exit.")
    print("==========================================")
    print(f"LLM: {ragchain.invoke('Hi')}")
    while True:
        query = input("You: ")
        if query.strip().lower() == "quit()":
            print("Exiting the chat. Goodbye!")
            break
        else:
            # Echo the user input for now. You can add more logic here.
            print(f"LLM: {ragchain.invoke(query)}")


if __name__ == "__main__":
    main()