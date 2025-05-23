# Langchain and Ollama examples

## Introduction

Examples of using Langchain and LangGraph with Ollama, Llama3.2 model and Gemma 3.
Ollama allows you to run open-source large language models, such as Llama 3.2, locally.
Learn more about Ollama integration on the LangChain [website](https://python.langchain.com/docs/integrations/chat/ollama/).

Tested with Python 3.10, Langchain 0.3.19 and LangGraph 0.3.8

* [Ollama](https://ollama.com/download) should be installed and running
* Pull a Llama 3.2 model to use with the library: `ollama pull llama3.2`





## Examples

### Basics

| **Section**                               | **What does it cover?**                                                                                        | **Code**                                                             |
|-------------------------------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| Prompts                                   | Invoke with user prompt, Invoke with system rules and user prompt, Invoke using HumanMessage and SystemMessage | [ollama_prompts.py](ollama_prompts.py)                               |
| JSON output                               | Invoke with JSON output                                                                                        | [ollama_json.py](ollama_json.py)                                     |
| Assistant                                 | Invoke with assistant message                                                                                  | [ollama_assistant.py](ollama_assistant.py)                           |
| Prompt template                           | Invoke with prompt template                                                                                    | [ollama_prompt_template.py](ollama_prompt_template.py)               |
| ResponseSchema and StructuredOutputParser | Invoke with prompt template and schema, return Python dictionary with parser                                   | [ollama_output_parser.py](ollama_output_parser.py)                   |
| Filter messages                           | Filter messages by type, name, id, during chain invoke                                                         | [ollama_filter_messages.py](ollama_filter_messages.py)               |
| Trim messages                             | Trim messages with token counter and with message count                                                        | [ollama_trim_messages.py](ollama_trim_messages.py)                   |
| Simple schema extractor                   | Create schema class and prompt as extractor. Parameters extraction from text.                                  | [ollama_schema_extractor.py](ollama_schema_extractor.py)             |
| Sequential chain                          | Invoke Sequential chain made by using 2 chains                                                                 | [ollama_basic_sequential_chain.py](ollama_basic_sequential_chain.py) |
| Structured output                         | Structured output with Pydantic, TypedDict and JSON. Required LLM with tools                                   | [ollama_structured_output.py](ollama_structured_output.py)           |

### Text loading and splitting

| **Section**                                  | **What does it cover?**                                                                                       | **Code**                                                                           |
|----------------------------------------------|---------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Basic document loaders for PDF, Word and txt | Basic document loaders for PDF, Word and txt                                                                  | [ollama_document_loader_basic.py](ollama_document_loader_basic.py)                 |
| Basic text splitter                          | Text splitting with CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter, Spacy, and NLTK | [ollama_text_splitter_basic.py](ollama_text_splitter_basic.py)                     |
| Basic chain                                  | Invoke basic chain with and without StrOutputParser                                                           | [ollama_chain_basic.py](ollama_chain_basic.py)                                     |

### Database

| **Section**                                  | **What does it cover?**                                                                                        | **Code**                                                                           |
|----------------------------------------------|----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Vector database ChromaDB                     | Simple document splitting and embedding to ChromaDB and retrieval with prompt                                  | [ollama_chromadb_basic.py](ollama_chromadb_basic.py)                               |
| Vector database FAISS                        | Simple document splitting and embedding to FAISS and retrieval with prompt                                     | [ollama_faiss_basic.py](ollama_faiss_basic.py)                                     |
| SQL query                                    | Basic query to sql database                                                                                    | [ollama_sql_basic.py](ollama_sql_basic.py)                                         |
| DuckDB SQL query                             | Import CSV file to DuckDB and create SQL query                                                                 | [ollama_duckdb.py](ollama_duckdb.py)                                               |
| Pinecone basics                              | Connect to Pinecone, create index, add documents to Pinecone, invoke quries to Pinecone                        | [ollama_pinecone_basic.py](ollama_pinecone_basic.py)                               |

### Agent

| **Section**                                  | **What does it cover?**                                                                                        | **Code**                                                                           |
|----------------------------------------------|----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Langchain agent                              | Invoke langchain agent to calculate ElU, Softplus and SoftMax                                                  | [ollama_agent_langchain.py](ollama_agent_langchain.py)                             |
| Langgraph agent                              | Invoke langgraph agent to calculate ElU, Softplus and SoftMax                                                  | [ollama_agent_langgraph.py](ollama_agent_langgraph.py)                             |
| Agent with SQL memory                        | Invoke langchain agent to chat and save memory to SQL database                                                 | [ollama_agent_sql_message_history.py](ollama_agent_sql_message_history.py)         |
| Agent with MongoDB memory                    | Invoke langchain agent to chat and save memory to MongoDB database                                             | [ollama_agent_mongodb_message_history.py](ollama_agent_mongodb_message_history.py) |


### Chat
| **Section**                                  | **What does it cover?**                                                                                        | **Code**                                                                           |
|----------------------------------------------|----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Chatbot with Gemma 3                         | Basic Chatbot with MemorySaver and Gemma 3                                                                     | [ollama_chatbot_basic.py](ollama_chatbot_basic.py)                                 |
| Chatbot with Langchain agent                 | Basic Chatbot with Langchain agent and cross_entropy_loss tool calling                                         | [ollama_chatbot_agent_langchain.py](ollama_chatbot_agent_langchain.py)             |
| Chatbot with Langgraph agent                 | Basic Chatbot with Langgraph agent and cross_entropy_loss tool calling                                         | [ollama_chatbot_agent_langgraph.py](ollama_chatbot_agent_langgraph.py)             |


### Router

| **Section**                                  | **What does it cover?**                                                                                        | **Code**                                                                           |
|----------------------------------------------|----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Semantic prompt router                       | Semantic routing between prompts by using RunnablePassthrough, RunnableLambda and cosine_similarity            | [ollama_router_semantic.py](ollama_router_semantic.py)               |
| Router with RunnableLambda                   | Create router to select between 3 sub-chains by using RunnableLambda                                           | [ollama_router_runnablelambda.py](ollama_router_runnablelambda.py)                 |
| Router with RunnableBranch                   | Create router to select between 3 sub-chains by using RunnableBranch                                           | [ollama_router_runnablebranch.py](ollama_router_runnablebranch.py)                 |

### Multimodal

| **Section**                                  | **What does it cover?**                                                                                        | **Code**                                                                           |
|----------------------------------------------|----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Multi-modal image prompting with Llava       | Simple prompt to image with multi-modal model Llava                                                            | [ollama_vision_llava_basic.py](ollama_vision_llava_basic.py)                       |
| Multi-modal image prompting with Gemma 3     | Simple prompt to image with multi-modal model Gemma 3                                                          | [ollama_vision_gemma3_basic.py](ollama_vision_gemma3_basic.py)                     |

### Debug

| **Section**      | **What does it cover?**      | **Code**                                                 |
|------------------|------------------------------|----------------------------------------------------------|
| Inspect runnable | Create chain and print graph | [ollama_inspect_runnable.py](ollama_inspect_runnable.py) |
