from langchain_ollama import ChatOllama
from langchain_community.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import OllamaEmbeddings


def semantic_prompt_router(data):
    query_embedding = embeddings.embed_query(data["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    print("Using Pytorch" if most_similar == pytorch_template else "Using Python ")
    return PromptTemplate.from_template(most_similar)


if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    python_template = """You are a Python expert.
    You are great at answering questions about Python in a concise and easy to understand manner.
    Here is a question:  {query}"""

    pytorch_template = """You are a Pytorch expert. You are great at answering Pytorch questions.
    Here is a question:  {query}"""

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    prompt_templates = [python_template, pytorch_template]
    # Create embeddings
    prompt_embeddings = embeddings.embed_documents(prompt_templates)
    # Create chain
    chain = (
            {"query": RunnablePassthrough()}
            | RunnableLambda(semantic_prompt_router)
            | llm_ollama
            | StrOutputParser()
    )
    # Python question
    msg_0 = chain.invoke("What's a decorator")
    print(f"Invoke chain with Python question -------------------------------------------------------------\n{msg_0}\n")
    # Pytorch question
    msg_1 = chain.invoke("What's a torch.nn.GELU")
    print(f"Invoke chain with Pytorch question ------------------------------------------------------------\n{msg_1}\n")
