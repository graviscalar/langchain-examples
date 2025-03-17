import numpy as np
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool


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
    # declare memory implementation of chat message history
    memory = InMemoryChatMessageHistory()
    # Create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    # Register tools
    tools = [cross_entropy_loss]
    # Create agent
    agent = create_tool_calling_agent(llm_ollama, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    config = {"configurable": {"session_id": "test-session"}}
    print(
        agent_with_chat_history.invoke(
            {"input": "Hi, I'm Alex! What's the output of cross_entropy_loss([10, 5, 3, 1, 4], [1, 0, 0, 0, 0])?"}, config
        )["output"]
    )
    print("-----------------------------------------------------------------------------------------------------------")
    print(agent_with_chat_history.invoke({"input": "Remember my name?"}, config)["output"])
    print("-----------------------------------------------------------------------------------------------------------")

    # Start chat
    while True:
        user_input = input('Chat with history: ')
        if user_input == "quit":
            break
        print(agent_with_chat_history.invoke({"input": user_input}, config)["output"])
        print("-------------------------------------------------------------------------------------------------------")
