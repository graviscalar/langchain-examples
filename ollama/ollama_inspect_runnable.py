from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import ollama



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
    # Create FAISS db
    vector_db = FAISS.from_documents(r_split, OllamaEmbeddings(model="nomic-embed-text"))

    retriever = vector_db.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm_ollama
            | StrOutputParser()
    )
    print(f"Print graph --------------------------------------------------------------------------------------------\n")
    chain.get_graph().print_ascii()

