from langchain_core.prompts import ChatPromptTemplate

def zero_shot_prompt():
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based only on the provided context:
        <context>
        {context}
        </context>

        Question: {input}
        """)
    return prompt
