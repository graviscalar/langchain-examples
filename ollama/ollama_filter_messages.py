import pprint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, filter_messages
from langchain_ollama import ChatOllama

if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    # Set up format to JSON mode
    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    messages = [
        SystemMessage("You are a PyTorch expert", id="1"),
        HumanMessage("nn.Module", id="2", name="Alex"),
        AIMessage("Base class for all neural network modules.", id="3", name="pytorch_assistant"),
        HumanMessage("What is the purpose of torch.autograd?", id="4", name="Viktor"),
        AIMessage("torch.autograd provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions.", id="5", name="Yulia"),
    ]
    # Filter messages by message type HumanMessage
    result = filter_messages(messages, include_types="human")
    print(f" Filter messages by message type HumanMessage ----------------------------------------------------------\n")
    pprint.pp(result)
    # Filter messages by names
    result = filter_messages(messages, exclude_names=["Alex", "pytorch_assistant"])
    print(f" Filter messages by names ------------------------------------------------------------------------------\n")
    pprint.pp(result)
    # Filter messages by ID
    result = filter_messages(messages, include_types=[HumanMessage, AIMessage], exclude_ids=["4", "5"])
    print(f" Filter messages by ID ---------------------------------------------------------------------------------\n")
    pprint.pp(result)
    # Filter messages in chain
    filter_ = filter_messages(exclude_names=["Yulia"])
    chain = filter_ | llm_ollama
    msg_txt = chain.invoke(messages)
    print(f"Filter messages in chain ------------------------------------------------------------\n{msg_txt.content}\n")
