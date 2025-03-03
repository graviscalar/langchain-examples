from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

if __name__ == "__main__":
    # Create loader for Txt file
    loader = TextLoader("../data/docs/quantization.txt")
    pages_txt = loader.load()
    # Invoke with prompt template
    # Create template
    template_txt = """Give me a short summary of {text}."""
    # Create a chat prompt template from a template string
    chat_template_txt = ChatPromptTemplate.from_template(template_txt)
    text = pages_txt[0].page_content

    # Create char splitter
    c_splitter = CharacterTextSplitter(chunk_size=512,
                                       chunk_overlap=64)
    c_split = c_splitter.split_text(text=text)
    print(f"Length of c_split = {len(c_split)}\n")
    print(f"Split Txt file with CharacterTextSplitter ---------------------------------------------------\n{c_split}\n")

    # Create recursive splitter
    r_splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                                chunk_overlap=64)
    r_split = r_splitter.split_text(text=text)
    print(f"Length of r_split = {len(r_split)}\n")
    print(f"Split Txt file with RecursiveCharacterTextSplitter ------------------------------------------\n{r_split}\n")
