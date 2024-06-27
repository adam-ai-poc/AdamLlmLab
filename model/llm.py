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
class LLM():
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
class OpenaiLLM(LLM):

    llm_config = LLM_CONFIG

    def __init__(self, llm_config, debug=False):
        self.openai_config: dict = llm_config["openAI"]
        self.model_name: str = self.openai_config["model_name"]
        self.max_tokens: str = self.openai_config["max_tokens"]
        self.temperature: str = self.openai_config["temperature"]
        super().__init__(os.environ.get("OPENAI_API_KEY"), self.model_name, self.max_tokens, self.temperature, debug, callback = get_openai_callback())