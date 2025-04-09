# BEFORE USE
# download Spacy model by:
# python -m spacy download en_core_web_sm

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_text_splitters import SpacyTextSplitter
from langchain_text_splitters import NLTKTextSplitter

if __name__ == "__main__":
    # Create loader for Txt file
    loader = TextLoader("../data/docs/quantization.txt")
    pages_txt = loader.load()
    text = pages_txt[0].page_content

    # Create char splitter
    c_splitter = CharacterTextSplitter(chunk_size=512,
                                       chunk_overlap=64)
    # Split text
    c_split = c_splitter.split_text(text=text)
    print(f"Length of c_split = {len(c_split)}\n")
    print(f"Split Txt file with CharacterTextSplitter ---------------------------------------------------\n{c_split}\n")

    # Create recursive splitter
    r_splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                                chunk_overlap=64)
    # Split text
    r_split = r_splitter.split_text(text=text)
    print(f"Length of r_split = {len(r_split)}\n")
    print(f"Split Txt file with RecursiveCharacterTextSplitter ------------------------------------------\n{r_split}\n")

    # Split document
    doc_split = r_splitter.split_documents(documents=pages_txt)
    print(f"Length of doc_split = {len(doc_split)}\n")
    print(f"Split Txt file with split_documents -------------------------------------------------------\n{doc_split}\n")

    # Create token splitter
    t_splitter = TokenTextSplitter(chunk_size=512,
                                   chunk_overlap=64)

    t_split = t_splitter.split_documents(documents=pages_txt)
    print(f"Length of t_split = {len(t_split)}\n")
    print(f"Split Txt file with TokenTextSplitter ------------------------------------------------------\n{t_split}\n")

    # Split with Spacy
    with open("../data/docs/quantization.txt") as f:
        quantization = f.read()
    # download Spacy model by:
    # python -m spacy download en_core_web_sm

    spacy_splitter = SpacyTextSplitter(chunk_size=512)
    spacy_split = spacy_splitter.split_text(quantization)
    print(f"Length of t_split = {len(spacy_split)}\n")
    print(f"Split Txt file with Spacy -------------------------------------------------------------------\n{t_split}\n")

    # Split with NTLK
    with open("../data/docs/quantization.txt") as f:
        quantization = f.read()

    nltk_splitter = NLTKTextSplitter(chunk_size=512)
    nltk_split = nltk_splitter.split_text(quantization)
    print(f"Length of t_split = {len(nltk_split)}\n")
    print(f"Split Txt file with NLTK --------------------------------------------------------------------\n{t_split}\n")
