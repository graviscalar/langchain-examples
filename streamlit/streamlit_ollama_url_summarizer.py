import streamlit as st
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
import validators


# To view this Streamlit app on a browser, run it with the following:
# streamlit run streamlit_ollama_url_summarizer.py

def st_app(llm):
    """ Simple URL summarizer with Ollama and Llama 3.2

    :return: None
    """
    st.set_page_config(page_title="Summarize URL with Llama 3.2")
    st.subheader('Summarize WEB address')

    url = st.text_input("WEB address", label_visibility="collapsed")

    if st.button("Summarize"):
        # Validate inputs
        if not validators.url(url) or "youtube.com" in url:
            st.error("Please enter a valid WEB address.")
        else:
            try:
                with st.spinner('Please wait...'):
                    # Load URL data
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    data = loader.load()
                    # Create prompt
                    prompt_template = """Write a summary of the: {text} """
                    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                    # Create chain
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    # Invoke chain
                    summary = chain.run(data)
                    # Display result of invoke
                    st.success(summary)
            except Exception as e:
                st.exception(f"An error occurred: {e}")


if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2:latest"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    st_app(llm=llm_ollama)
