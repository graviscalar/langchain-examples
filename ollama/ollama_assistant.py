from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


def user_assistant_invoke(llm, messages: list) -> str:
    """
    Invoke list of massages

    :param llm: ChatOllama Model
    :param messages: list of messages
    :return: LLM response
    """

    ai_msg = llm.invoke(messages)
    return ai_msg.content


if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    # Set up format to JSON mode
    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    # Sending messages in OPenAI and Llama format
    openai_llama_messages = [
        {
            "role": "user",
            "content": "I like Python and Pytorch?",
        },
        {
            "role": "assistant",
            "content": "That's great! Python and PyTorch are a powerful combination for machine learning and deep learning.",
        },
        {
            "role": "user",
            "content": "From my previous messages, can you identify any topics or technologies that I seem to be interested in?",
        }
    ]
    msg_0 = user_assistant_invoke(llm=llm_ollama, messages=openai_llama_messages)
    print(f"Invoke messages in OPenAI and Llama format-----------------------------------------------------\n{msg_0}\n")

    # Sending messages in Langchain format
    langchain_messages = [
        HumanMessage(content="I like Python and Pytorch?"),
        AIMessage(content="That's great! Python and PyTorch are a powerful combination for machine learning and deep learning."),
        HumanMessage(content="From my previous messages, can you identify any topics or technologies that I seem to be interested in?")
    ]
    msg_1 = user_assistant_invoke(llm=llm_ollama, messages=langchain_messages)
    print(f"Invoke messages in Langchain format -----------------------------------------------------------\n{msg_1}\n")
