# Adam's Lab
Welcome to Adam's Lab, a repository designed to streamline the AI engineer's experience in experimenting with and exploring the capabilities and applications of Large Language Models (LLMs). This repository serves as a foundation for Proof-of-Concept (POC) and pathfinding initiatives aimed at leveraging LLM techniques to advance understanding in this domain. The classes, functions, and variables used in this codebase are designed to be as generic as possible, facilitating easy adaptation to a wide range of specific use cases. While primarily leveraging the Langchain ecosystem, this repository is also designed to integrate seamlessly with other ecosystems. It provides a versatile framework for experimenting with and implementing LLM techniques and concepts.



## Purpose
The primary goal of this repository is to learn, explore and experiment with various techniques and concepts related to Large Language Models (LLMs). It focuses on three main applications:

1. Retrieval Augmentation Generation (RAG): Investigating methods to enhance information retrieval and generation capabilities using LLMs.

2. Fine-tuning: Exploring strategies for fine-tuning pre-trained LLMs to adapt them to specific tasks or domains.

3. Agents for Tool Calling: Developing agents that interact with tools or systems based on natural language inputs, showcasing LLMs' potential in task automation and integration.

## Get Started
### Prerequisites
#### Skills
1. Python
2. Object Oriented Programming
3. Git
4. Basic understanding of LLMs and their applications
   
(optional)
6. Frontend - react.js, css, html
7. Devops - github actions, cicd

#### Setup 
1. Generate openai api key: "sk-xxx"
2. Anaconda/Miniconda installed

### Environment Setup
1. Create a conda environment:

    `conda create -n adamlab python=3.11`

2. Activate the conda environment:

    `conda activate adamlab`

3. Install dependencies. In the base directory:

    `pip install -r requirements.txt`

4. Setup Openai Api Key:

    - Create a file in the base directory named `.env` and set the environment variable:

    `OPENIAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxx"`
    
    - Replace the key with the actual key which you generated from OpenAI's web portal.

5. Test the terminal-based rag app `app/rag_app.py`.

    - After activating your conda environment, type the command in the terminal in the base directory to bring up the terminal-based chat interface.

    `python app/rag_app.py`

    - Use the `--help` flag to see all the flags available for the script.

6. To run the streamlit-based webapp, type this command in your terminal at the base directory:

    `run.bat`

    - Your webapp should be hosed at localhost:8501 (default). To stop the service, type Ctrl+C in the terminal.
