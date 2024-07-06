from .retrieve import *
from . import ADAM_RETRIEVER
from model import ADAM_LLM, ADAM_PROMPT
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

'''
Chain class to link the rag pipelines
'''
class RagChain:

    def __init__(self, retriever=None, system_prompt=None, llm=None, debug=False, **kwargs):
        self.debug = debug

        # Retriever initialization
        self.retriever = self.set_component(component=retriever, ADAM_COMPONENT=ADAM_RETRIEVER)
        print(f"Retriever: {self.retriever} Initialized.")

        self.set_system_prompt(system_prompt=system_prompt)
        print("System prompt initialized")

        self.llm = self.set_component(component=llm, ADAM_COMPONENT=ADAM_LLM)
        print(f"LLM: {self.llm.model_name} Initialized.")

        print(self.get_config_string()) if self.debug else None

        self.chain = self.create_chain(debug=self.debug)

    def __call__(self, query):
        return self.chain.invoke(query)
    
    def stream(self, query):
        return self.chain.stream(query)

    def get_config_string(self):
        return f"""
            ==========================================
            Ragchain configurations
            ------------------------------------------
            Retriever: {self.retriever}
            System prompt: {self.system_prompt}
            LLM: {self.llm.model_name}
            ==========================================
        """

    def create_chain(self, debug=False):
        if debug==True:
            import langchain
            langchain.debug = True

        chain = ({"context": self.retriever.retriever | self.format_docs, "input": RunnablePassthrough()} \
            | self.system_prompt \
            | self.llm.model \
            | StrOutputParser())
        
        return chain

    # Preprocessing
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def set_component(self, component, ADAM_COMPONENT=None):
        if isinstance(component, dict):
            return self._init_component_from_dict(ADAM_COMPONENT, component)
        else:
            return component
    
    def set_system_prompt(self, system_prompt):
        if isinstance(system_prompt, dict):
            prompt_type = system_prompt["prompt_type"]
            self.system_prompt = ADAM_PROMPT[prompt_type]
        else:
            self.system_prompt = system_prompt

    def _init_component_from_dict(self, COMPONENT, component_cfg:dict):
        component = next(iter(component_cfg))
        component_parameters_cfg = component_cfg[component]
        return COMPONENT[component](**component_parameters_cfg, debug = self.debug)
