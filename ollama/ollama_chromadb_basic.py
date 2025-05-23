from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    # Pull embeddings model. More info here https://ollama.com/library/nomic-embed-text
    ollama.pull("nomic-embed-text")
    print(f"Pull for embedding model finished")

    # Create loader for Txt file
    loader = TextLoader("../data/docs/tensor_views.txt")
    pages_txt = loader.load()
    text = pages_txt[0].page_content

    # Create recursive splitter
    r_splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                                chunk_overlap=64)
    r_split = r_splitter.split_documents(pages_txt)
    # Create vector database
    vector_db = Chroma.from_documents(
        documents=r_split,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="simple-rag",
    )
    print(f"Vector db created")
    # Query by turning into retriever
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 1, "fetch_k": 5}
    )
    docs = retriever.invoke("Tell me about view()")
    for doc in docs:
        print(doc.page_content)
