import pprint
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
import numpy as np


@tool
def elu(x: float, alpha: float = 1.0) -> float:
    """ Applies the Exponential Linear Unit (ELU) function.

    Args:
        x: input value.
        alpha: value for the ELU formulation. Default: 1.0
    """
    if x > 0:
        return x
    else:
        return alpha * (np.exp(x) - 1)


@tool
def softplus(x: float, beta: float = 1.0) -> float:
    """ Applies the Softplus function.

    Args:
        x: input value.
        beta: value for the ELU formulation. Default: 1.0
    """
    return np.log(1 + np.exp(x * beta)) / beta


@tool
def softmax(x):
    """ Applies the softmax function.

    Args:
        x: input array.
    """
    exp_values = np.exp(x)
    exp_values_sum = np.sum(exp_values)
    return exp_values / exp_values_sum


if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)
    # register tools
    tools = [elu, softplus, softmax]

    # Create agent
    langgraph_agent_executor = create_react_agent(llm_ollama, tools)

    query = "Calculate elu(-3)"
    # Invoke agent
    messages = langgraph_agent_executor.invoke({"messages": [("human", query)]})
    print(f"Invoke agent to calculate ELU --------------------------------------------------------------------------\n")
    pprint.pp(messages["messages"][-1].content)

    query = "Calculate softplus(-3)"
    # Invoke agent
    messages = langgraph_agent_executor.invoke({"messages": [("human", query)]})
    print(f"Invoke agent to calculate Softplus ---------------------------------------------------------------------\n")
    pprint.pp(messages["messages"][-1].content)

    query = "Calculate softmax([2, 4, 5, 3])"
    # Invoke agent
    messages = langgraph_agent_executor.invoke({"messages": [("human", query)]})
    print(f"Invoke agent to calculate Softmax ----------------------------------------------------------------------\n")
    pprint.pp(messages["messages"][-1].content)
