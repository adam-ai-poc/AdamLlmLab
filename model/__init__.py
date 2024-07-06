from .llm import OpenaiLLM
from .prompt import zero_shot_prompt

ADAM_LLM = {
    "openai": OpenaiLLM
}

ADAM_PROMPT = {
    "zero-shot": zero_shot_prompt()
}