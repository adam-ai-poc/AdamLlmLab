import os
import json

from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from .utils import read_config

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
LLM_CONFIG = read_config(os.path.join(os.path.dirname(__file__), "config.yaml"), "llm_config")

'''
Base Agemt class
'''
class Agent():
    def __init__(self, api_key, model_name, max_tokens, temperature, debug, callback=None):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.debug = debug
        self.callback = callback
        self.llm = ChatOpenAI(model=model_name, max_tokens=max_tokens, temperature=temperature)

    def __call__(self, query):
        
        # If debug is turned on, print metadata as well as pricing
        if self.debug:
            print(f"Query: {query}")

            print(f"Usage metadata: ")
            if self.callback:
                with self.callback as callback:
                    response = self.llm.invoke(query)
                    print(callback)
            else:
                print(json.dumps(response.usage_metadata, indent=1))

            print(f"Response metadata: ")
            print(json.dumps(response.response_metadata, indent=1))

        else:
            response = self.llm.invoke(query)
        
        return response.content

'''
OpenAI backend service 
'''
class OpenaiAgent(Agent):

    openai_config: dict = LLM_CONFIG["openAI"]
    model_name: str = openai_config["model_name"]
    max_tokens: str = openai_config["max_tokens"]
    temperature: str = openai_config["temperature"]

    def __init__(self, api_key=OPENAI_API_KEY, model_name=model_name, max_tokens=max_tokens, temperature=temperature, debug=False, **kwargs):
        super().__init__(api_key, model_name, max_tokens, temperature, debug, callback = get_openai_callback())