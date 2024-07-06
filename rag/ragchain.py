from .retrieve import *
from . import ADAM_RETRIEVER
from model import ADAM_LLM, ADAM_PROMPT
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

'''
Chain class to link the rag pipelines
'''
class RagChain:

    def __init__(self, num_history=3, retriever=None, system_prompt=None, llm=None, debug=False, **kwargs):

        self.chat_history = []
        self.num_history = num_history
        self.debug = debug

        # Retriever initialization
        self.retriever = self.set_component(component=retriever, ADAM_COMPONENT=ADAM_RETRIEVER)
        print(f"Retriever: {self.retriever} Initialized.") if self.debug else None

        self.set_system_prompt(system_prompt=system_prompt)
        print("System prompt initialized") if self.debug else None

        self.llm = self.set_component(component=llm, ADAM_COMPONENT=ADAM_LLM)
        print(f"LLM: {self.llm.model_name} Initialized.") if self.debug else None

        print(self.get_config_string()) if self.debug else None

        self.chain = self.create_chain(debug=self.debug)

    def __call__(self, query):
        recent_history = self.chat_history[-(self.num_history*2):]
        print("Recent history: ", recent_history) if self.debug else None
        response = self.chain.invoke(          
            {"context": self.get_contexts_from_query(query), \
            "question": query, \
            "chat_history": recent_history}
            )
        self.update_chat_history(query, response)
        return response
    
    def get_contexts_from_query(self, query):
        contexts = self.retriever.retrieve(query)
        formatted_contexts = self.format_docs(contexts)
        print("Contexts: ", formatted_contexts) if self.debug else None
        return formatted_contexts
    
    def update_chat_history(self, query, response):
        human_query = HumanMessage(query)
        ai_response = AIMessage(response)
        self.chat_history.append(human_query)
        self.chat_history.append(ai_response)
    
    def stream(self, query):
        recent_history = self.chat_history[-(self.num_history*2):]
        print("Recent history: ", recent_history) if self.debug else None
        
        streamer = self.chain.stream(          
            {"context": self.get_contexts_from_query(query), \
            "question": query, \
            "chat_history": recent_history}
            )
        
        # Stream response using yield
        response = ""
        for chunk in streamer:
            response += chunk
            yield chunk
        self.update_chat_history(query, response)

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

        chain = (
            self.system_prompt \
            | self.llm.model \
            | StrOutputParser()
            )
        
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
