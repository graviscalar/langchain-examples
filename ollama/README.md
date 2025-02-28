# Langchain and Ollama examples

## Introduction

Examples of using Langchain with Ollama and Llama3.2 model.
Ollama allows you to run open-source large language models, such as Llama 3.2, locally.
Learn more about Ollama integration on the LangChain [website](https://python.langchain.com/docs/integrations/chat/ollama/).

Tested with Python 3.10

* [Ollama](https://ollama.com/download) should be installed and running
* Pull a Llama 3.2 model to use with the library: `ollama pull llama3.2`





## Examples
| **Section** | **What does it cover?**                                                                                        | **Code**                               |
|-------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------|
| Prompts     | Invoke with user prompt, Invoke with system rules and user prompt, Invoke using HumanMessage and SystemMessage | [ollama_prompts.py](ollama_prompts.py) |
| JSON output | Invoke with JSON output                                                                                        | [ollama_json.py](ollama_json.py)       |
| assistant   | Invoke with assistant mesage                                                                                   | [ollama_assistant.py](ollama_assistant.py)

