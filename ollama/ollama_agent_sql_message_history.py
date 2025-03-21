from langchain.chains import create_sql_query_chain
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
import numpy as np
import uuid
import sqlite3

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
    # Create session ID
    session_uuid = str(uuid.uuid1())
    # DB name
    sql_db_name = 'memory.db'
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

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
    # Create SQL memory
    sql_message_history = SQLChatMessageHistory(
        session_id=session_uuid, connection='sqlite:///' + sql_db_name
    )
    # Create agent memory
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: sql_message_history,
        input_messages_key="input",
        history_messages_key="chat_history",

    )
    # Create config
    config = {"configurable": {"session_id": sql_message_history}}
    # Start chat
    while True:
        user_input = input('Chat with history: ')
        if user_input == "quit":
            break
        print(agent_with_chat_history.invoke({"input": user_input}, config)["output"])
        print("-------------------------------------------------------------------------------------------------------")
    # Print all records from database
    conn = sqlite3.connect(sql_db_name)
    cur = conn.cursor()
    cur.execute('SELECT * FROM message_store')
    rows = cur.fetchall()
    conn.close()
    for row in rows:
        print(row)