from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


# Define the function that calls the model
def call_model(state: MessagesState):
    response = llm_ollama.invoke(state["messages"])
    return {"messages": response}


if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "gemma3:4b"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    # Define the (single) node in the graph
    workflow.add_edge(START, "gemma3")
    workflow.add_node("gemma3", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # create a config that we pass into the runnable every time
    config = {"configurable": {"thread_id": "gemma123"}}

    # Start chat
    while True:
        user_input = input('Chat with history: ')

        input_messages = [HumanMessage(user_input)]
        output = app.invoke({"messages": input_messages}, config)
        output["messages"][-1].pretty_print()
