from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
import pprint


def prompt_template_invoke(llm, prompt_template) -> str:
    """
    Invoke with prompt template

    :param llm: ChatOllama Model
    :param prompt_template: prompt template
    :return ai_msg: LLM response
    """

    ai_msg = llm.invoke(prompt_template)
    return ai_msg.content


if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    # Create loader for PDF file
    loader = PyPDFLoader("../data/docs/pytorch_moduledict.pdf")
    pages_pdf = loader.load()
    # Invoke with prompt template
    # Create template
    template_pdf = """Extract a keywords from {text}."""
    # Create a chat prompt template from a template string
    chat_template_pdf = ChatPromptTemplate.from_template(template_pdf)
    text = pages_pdf[0].page_content
    # Format the chat template into a list of finalized messages
    ollama_template_pdf = chat_template_pdf.format_messages(text=text)

    msg_pdf = prompt_template_invoke(llm=llm_ollama, prompt_template=ollama_template_pdf)
    print(f"Extract keywords from PDF file --------------------------------------------------------------\n{msg_pdf}\n")

    # Create loader wor Word file
    loader = Docx2txtLoader("../data/docs/pytorch_sequential.docx")
    pages_doc = loader.load()
    # Print metadata
    # pprint.pp(pages_doc[0].metadata)
    # Print content
    # print(pages_doc[0].page_content)
    # Invoke with prompt template
    # Create template
    template_doc = """Identify the main topic of {text}."""
    # Create a chat prompt template from a template string
    chat_template_doc = ChatPromptTemplate.from_template(template_doc)
    text = pages_doc[0].page_content
    # Format the chat template into a list of finalized messages
    ollama_template_doc = chat_template_doc.format_messages(text=text)

    msg_doc = prompt_template_invoke(llm=llm_ollama, prompt_template=ollama_template_doc)
    print(f"Identify the main topic from Doc file -------------------------------------------------------\n{msg_doc}\n")

    # Create loader wor Txt file
    loader = TextLoader("../data/docs/pytorch_parameterdict.txt")
    pages_txt = loader.load()
    # Invoke with prompt template
    # Create template
    template_txt = """Give me a short summary of {text}."""
    # Create a chat prompt template from a template string
    chat_template_txt = ChatPromptTemplate.from_template(template_txt)
    text = pages_txt[0].page_content
    # Format the chat template into a list of finalized messages
    ollama_template_txt = chat_template_txt.format_messages(text=text)

    msg_txt = prompt_template_invoke(llm=llm_ollama, prompt_template=ollama_template_txt)
    print(f"Short summary of Txt file -------------------------------------------------------------------\n{msg_txt}\n")
