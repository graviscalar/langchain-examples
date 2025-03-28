from langchain_ollama import ChatOllama
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
import time
from uuid import uuid4

if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    # use your Pinecone API key to initialize your client
    pinecone_api_key = "OBTAIN-Pinecone-API-KEY-BY-CREATING-FREE-ACCOUNT"
    # Initialize
    pc = Pinecone(api_key=pinecone_api_key)

    # Create a serverless index name
    index_name = "ollama-pinecone-basic"
    # Create a serverless index or verify existing
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,  # Replace with your model dimensions, 768 is nomic-embed-text size
            metric="cosine",  # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)


    # Create Pinecone vector store integration
    index = pc.Index(index_name)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Load documents
    docs = []
    loader = TextLoader("../data/docs/pytorch_parameterdict.txt")
    pages_txt = loader.load()
    docs.append(pages_txt[0])
    loader = TextLoader("../data/docs/quantization.txt")
    pages_txt = loader.load()
    docs.append(pages_txt[0])
    loader = TextLoader("../data/docs/tensor_views.txt")
    pages_txt = loader.load()
    docs.append(pages_txt[0])

    uuids = [str(uuid4()) for _ in range(len(docs))]

    # Add or update documents in the vectorstore.
    vector_store.add_documents(documents=docs, ids=uuids)

    # Simple similarity search
    print(f"Simple similarity search -------------------------------------------------------------------------------\n")
    # Return pinecone documents most similar to query.
    results = vector_store.similarity_search(
        "Tensor views",
        k=2
    )
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")

    # Similarity search with score
    print(f"Simple similarity search with score --------------------------------------------------------------------\n")
    # Return pinecone documents most similar to query, along with scores.
    results = vector_store.similarity_search_with_score(
        "nn.Parameter", k=2
    )
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
