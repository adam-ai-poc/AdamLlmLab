from langchain_core.prompts import ChatPromptTemplate

def zero_shot_prompt():
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based only on the provided context:

        <Chat history>
        {chat_history}
        </Chat history>

        <Context>
        {context}
        </Context>

        Question: {question}
        """)
    return prompt
