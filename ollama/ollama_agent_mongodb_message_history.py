from langchain.chains import create_sql_query_chain
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
import numpy as np
import uuid
import pymongo

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
    mongo_db_name = "db_chat"

    # DB collection name
    mongo_db_collection = "chat_histories"

    # mongodb connection
    mongo_db_connection = "mongodb://localhost:27017"

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

    # Create mongodb memory
    chat_message_history = MongoDBChatMessageHistory(
        session_id=session_uuid,
        connection_string=mongo_db_connection,
        database_name=mongo_db_name,
        collection_name=mongo_db_collection,
    )

    # Create agent memory
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: chat_message_history,
        input_messages_key="input",
        history_messages_key="chat_history",

    )

    # Create config
    config = {"configurable": {"session_id": session_uuid}}

   # Start chat
    while True:
        user_input = input('Chat with history: ')
        if user_input == "quit":
            break
        print(agent_with_chat_history.invoke({"input": user_input}, config)["output"])
        print("-------------------------------------------------------------------------------------------------------")

    # Print all records from mongodb database
    mdb_client = pymongo.MongoClient(mongo_db_connection)
    mydb = mdb_client[mongo_db_name]
    mdb_col = mydb[mongo_db_collection]

    for x in mdb_col.find():
        print(x)