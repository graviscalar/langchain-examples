from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


def user_invoke(llm, human_prompt: str = None) -> str:
    """
    Invoke with user prompt only

    :param llm: ChatOllama Model
    :param human_prompt: User prompt
    :return ai_msg: LLM response
    """

    ai_msg = llm.invoke(human_prompt)
    return ai_msg.content


def system_user_invoke(llm, system_rule: str = None, human_prompt: str = None) -> str:
    """
    Invoke with system rules and user prompt

    :param llm: ChatOllama Model
    :param human_prompt: User prompt
    :param system_rule: system rules
    :return ai_msg: LLM response
    """

    messages = [
        ("system", system_rule),
        ("user", human_prompt)
    ]

    ai_msg = llm.invoke(messages)
    return ai_msg.content


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

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    # Sending user prompt to LLM
    # Create your prompt in order to talk with LLM
    # For example: "I like programming in Python"
    user_prompt = "I like programming in Python"
    msg_0 = user_invoke(llm=llm_ollama, human_prompt=user_prompt)
    print(f"Sending user prompt to LLM --------------------------------------------------------------------\n{msg_0}\n")

    # Invoke with system rules and user prompt
    # Describe rules for system
    system_rules = "You are a helpful assistant."
    user_prompt = "I like programming in Python"
    msg_1 = system_user_invoke(llm=llm_ollama, system_rule=system_rules, human_prompt=user_prompt)
    print(f"Sending system rules and user prompt to LLM ---------------------------------------------------\n{msg_1}\n")

    # Invoke using HumanMessage and SystemMessage
    # Describe rules for system
    system_rules = "You are a helpful assistant."
    user_prompt = "I like programming in Python"
    msg_2 = system_user_message_invoke(llm=llm_ollama, system_rule=system_rules, human_prompt=user_prompt)
    print(f"Sending using HumanMessage and SystemMessage --------------------------------------------------\n{msg_2}\n")
