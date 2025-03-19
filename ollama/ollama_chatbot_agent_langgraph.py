import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage


def softmax(x):
    """ Applies the softmax function.

    Args:
        x: input array.
    """
    exp_values = np.exp(x)
    exp_values_sum = np.sum(exp_values)
    return exp_values / exp_values_sum


@tool
def cross_entropy_loss(y_pred: list, y_true: list):
    """ This criterion computes the cross entropy loss between input logits and target.

    Args:
        y_pred: predictions array.
        y_true: ground true array.
    """
    y_pred = softmax(y_pred)
    loss = 0

    for i in range(len(y_pred)):
        loss = loss + (-1 * y_true[i] * np.log(y_pred[i]))

    return loss


if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)
    # Register tools
    tools = [cross_entropy_loss]
    # System message
    system_message = SystemMessage(content="You are a helpful assistant.")
    # An in-memory checkpoint saver
    memory = MemorySaver()
    # Create agent
    langgraph_agent_executor = create_react_agent(
        llm_ollama, tools, prompt=system_message, checkpointer=memory
    )

    config = {"configurable": {"thread_id": "test-thread"}}
    print(
        langgraph_agent_executor.invoke(
            {
                "messages": [
                    ("user", "Hi, I'm Alex! What's the output of cross_entropy_loss([10, 5, 3, 1, 4], [1, 0, 0, 0, 0])?")
                ]
            },
            config,
        )["messages"][-1].content
    )
    print("-----------------------------------------------------------------------------------------------------------")
    print(
        langgraph_agent_executor.invoke(
            {"messages": [("user", "Remember my name?")]}, config
        )["messages"][-1].content
    )
    print("-----------------------------------------------------------------------------------------------------------")
    print(
        langgraph_agent_executor.invoke(
            {"messages": [("user", "what was that output again?")]}, config
        )["messages"][-1].content
    )
    # Start chat
    while True:
        user_input = input('Chat with history: ')
        if user_input == "quit":
            break
        print(
            langgraph_agent_executor.invoke(
                {"messages": [("user", user_input)]}, config
            )["messages"][-1].content
        )
        print("-------------------------------------------------------------------------------------------------------")
