# Langchain and Ollama examples

## Introduction

Examples of using Langchain with Ollama and Llama3.2 model.
Ollama allows you to run open-source large language models, such as Llama 3.2, locally.
Learn more about Ollama integration on the LangChain [website](https://python.langchain.com/docs/integrations/chat/ollama/).

Tested with Python 3.10 and Langchain 0.3.19

* [Ollama](https://ollama.com/download) should be installed and running
* Pull a Llama 3.2 model to use with the library: `ollama pull llama3.2`





## Examples
| **Section**                                  | **What does it cover?**                                                                                       | **Code**                                                             |
|----------------------------------------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| Prompts                                      | Invoke with user prompt, Invoke with system rules and user prompt, Invoke using HumanMessage and SystemMessage | [ollama_prompts.py](ollama_prompts.py)                               |
| JSON output                                  | Invoke with JSON output                                                                                       | [ollama_json.py](ollama_json.py)                                     |
| Assistant                                    | Invoke with assistant message                                                                                 | [ollama_assistant.py](ollama_assistant.py)                           |
| Prompt template                              | Invoke with prompt template                                                                                   | [ollama_prompt_template.py](ollama_prompt_template.py)               |
| ResponseSchema and StructuredOutputParser    | Invoke with prompt template and schema, return Python dictionary with parser                                  | [ollama_output_parser.py](ollama_output_parser.py)                   |
| Basic document loaders for PDF, Word and txt | Basic document loaders for PDF, Word and txt                                                                  | [ollama_document_loader_basic.py](ollama_document_loader_basic.py)   |
| Basic text splitter                          | Text splitting wiht CharacterTextSplitter, RecursiveCharacterTextSplitter and TokenTextSplitter               | [ollama_text_splitter_basic.py](ollama_text_splitter_basic.py)       |
| Basic chain                                  | Invoke basic chain with and without StrOutputParser                                                           | [ollama_chain_basic.py](ollama_chain_basic.py)                       |
| Sequential chain                             | Invoke Sequential chain made by using 2 chains                                                                | [ollama_basic_sequential_chain.py](ollama_basic_sequential_chain.py) |
| Vector database                              | Simple document splitting and embedding to ChromaDB and retrieval with prompt                                 | [ollama_chromadb_basic.py](ollama_chromadb_basic.py)                 |
| Multi-modal image prompting                  | Simple prompt to image with multi-modal model Llava                                                           ||
