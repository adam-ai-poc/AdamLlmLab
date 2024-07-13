import os
import json
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain_core.runnables import ConfigurableField
from rag.utils import read_config

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_CONFIG = read_config(os.path.join(os.path.dirname(__file__), "config.yaml"), "llm")

'''
Base Agemt class
'''
class LLM():
    def __init__(self, model, model_name, max_tokens, temperature, debug, callback=None):
        self.model = model
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.debug = debug
        self.callback = callback
        self.llm_configurations = self.set_llm_configurations()

    def __call__(self, query):
        """Get response from LLM."""
        self.set_llm_configurations()
        response = self.model.with_config(configurable=self.llm_configurations).invoke(query)
        
        return response
    
    def set_llm_configurations(self):
        self.llm_configurations = {
            "llm_temperature": self.temperature,
            "llm_max_tokens": self.max_tokens,
        }
        print("LLM Configurations: ", self.llm_configurations)
    
    def get(self, key: str):
        """Generic getter method to return components."""
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

'''
OpenAI backend class 
'''
class OpenaiLLM(LLM):

    # Default
    llm_config = LLM_CONFIG["openai"]
    model_name: str = llm_config["model_name"]
    max_tokens: str = llm_config["max_tokens"]
    temperature: str = llm_config["temperature"]

    def __init__(self, model_name=model_name, max_tokens=max_tokens, temperature=temperature, debug=False):
        self.model_name=model_name
        self.max_tokens=max_tokens
        self.temperature=temperature
        self.model = ChatOpenAI(model=model_name, max_tokens=max_tokens, temperature=temperature).configurable_fields(
            temperature=ConfigurableField(
                id="llm_temperature",
                name="LLM Temperature",
                description="The temperature of the LLM",
            ),
            max_tokens=ConfigurableField(
                id="llm_max_tokens",
                name="LLM Max Tokens",
                description="The Max Tokens of the LLM",
            ),
        )
        self.debug=debug
        super().__init__(self.model, self.model_name, self.max_tokens, self.temperature, debug, callback = get_openai_callback())