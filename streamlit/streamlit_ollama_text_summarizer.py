import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# To view this Streamlit app on a browser, run it with the following:
# streamlit run streamlit_ollama_text_summarizer.py

def st_app():
    """ Simple text summarizer with Ollama and Gemma 3

    :return: None
    """
    st.set_page_config(page_title="Summarize Text with Gemma 3")
    st.subheader('Summarize Text')

    source_text = st.text_area("Source Text", label_visibility="collapsed", height=270)

    # If the 'Summarize' button is clicked
    if st.button("Summarize"):
        # Validate inputs
        if not source_text.strip():
            st.error(f"Please provide the missing fields.")
        else:
            try:
                with st.spinner('Please wait...'):
                    # use your Ollama model. Verify loaded models with 'ollama list'
                    model = "gemma3:4b"
                    temperature = 0

                    llm_ollama = ChatOllama(model=model,
                                            temperature=temperature)
                    # Split the source text
                    text_splitter = CharacterTextSplitter()
                    texts = text_splitter.split_text(source_text)
                    # Create Document objects for the texts (max 3 pages)
                    docs = [Document(page_content=text) for text in texts]
                    # Define prompt
                    prompt = ChatPromptTemplate.from_messages(
                        [("system", "Write a concise summary of the following:\\n\\n{context}")]
                    )
                    # Instantiate chain
                    chain = create_stuff_documents_chain(llm_ollama, prompt)
                    # Invoke chain
                    result = chain.invoke({"context": docs})
                    # Display a success message.
                    st.success(result)
            except Exception as e:
                st.exception(f"An error occurred: {e}")

if __name__=="__main__":
    st_app()