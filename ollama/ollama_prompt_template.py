from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


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

    # Invoke with prompt template
    # Create template
    template = """Answer the user's question, '{text}', in a {style} style"""
    # Create a chat prompt template from a template string
    chat_template = ChatPromptTemplate.from_template(template)
    style = """Pytorch expert"""
    text = """Give me a brief overview of Module."""
    # Format the chat template into a list of finalized messages
    ollama_template = chat_template.format_messages(style=style, text=text)

    msg_0 = prompt_template_invoke(llm=llm_ollama, prompt_template=ollama_template)
    print(f"Invoke with prompt template -------------------------------------------------------------------\n{msg_0}\n")
