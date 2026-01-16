from langchain_core.prompts import PromptTemplate

def prompt_temp():
    prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
Using the context below, answer the question in a clear and friendly way.
You may rephrase the information, but do NOT hallucinate or add facts that are not in the context.

Context:
{context}

Question:
{question}

Answer in a friendly, natural style:"""
)
    return prompt