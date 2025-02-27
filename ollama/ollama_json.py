from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


def system_user_message_invoke(llm, system_rule: str = None, human_prompt: str = None) -> str:
    """
    Invoke using HumanMessage and SystemMessage

    :param llm: ChatOllama Model
    :param human_prompt: User prompt
    :param system_rule: system rules
    :return: LLM response
    """

    messages = [
        SystemMessage(content=system_rule),
        HumanMessage(content=human_prompt)
    ]

    ai_msg = llm.invoke(messages)
    return ai_msg.content


if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    # Set up format to JSON mode
    llm_ollama = ChatOllama(model=model,
                            temperature=temperature,
                            format="json")

    # Sending user prompt to LLM
    system_rules = "You are a PyTorch expert. Respond using JSON only."
    user_prompt = "Explain the softmax function and its use in machine learning."
    msg_0 = system_user_message_invoke(llm=llm_ollama, system_rule=system_rules, human_prompt=user_prompt)
    print(f"Invoke and create JSON in output --------------------------------------------------------------\n{msg_0}\n")

