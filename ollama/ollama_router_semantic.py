from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import OllamaEmbeddings
from langchain_community.utils.math import cosine_similarity

if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "gemma3:4b"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    # Python template
    python_template = """You are an expert in Python. \
    You are great at answering questions about Python language. \
    Always answer questions starting with "As Python expert". \
    Respond to the following question:
        
    Question: {question}
    Answer:"""

    # Pytorch template
    pytorch_template = """You are an expert in Pytorch. \
    You are great at answering questions about Pytorch framework. \
    Always answer questions starting with "As Pytorch expert". \
    Respond to the following question:
        
    Question: {question}
    Answer:"""

    # Scikit-learn template
    scikit_template = """You are an expert in Scikit-learn. \
    You are great at answering questions about Scikit-learn library. \
    Always answer questions starting with "As Scikit-learn expert". \
    Respond to the following question:
        
    Question: {question}
    Answer:"""

    # Create embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    prompt_templates = [python_template, pytorch_template, scikit_template]
    prompt_embeddings = embeddings.embed_documents(prompt_templates)


    def prompt_router(input):
        query_embedding = embeddings.embed_query(input["question"])
        similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
        most_similar = prompt_templates[similarity.argmax()]
        if most_similar == python_template:
            print("Using Python")
        elif most_similar == pytorch_template:
            print("Using Pytorch")
        else:
            print("Using Scikit-learn")
        return PromptTemplate.from_template(most_similar)


    # Create a final chain
    final_chain = (
            {"question": RunnablePassthrough()}
            | RunnableLambda(prompt_router)
            | llm_ollama
            | StrOutputParser()
    )

    # Route to Python chain
    result = final_chain.invoke("tell me about decorators")
    print(f"Routing to Python chain ----------------------------------------------------------------------\n{result}\n")

    # Route to Pytorch chain
    result = final_chain.invoke("tell me about torch.nn.Softmax")
    print(f"Routing to Pytorch chain ---------------------------------------------------------------------\n{result}\n")

    # Route to Scikit-learn chain
    result = final_chain.invoke("tell me about Gaussian process regression in scikit-learn")
    print(f"Routing to Scikit-learn chain ----------------------------------------------------------------\n{result}\n")
