import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate


# To view this Streamlit app on a browser, run it with the following:
# streamlit run streamlit_ollama_agent_search.py

@tool
def tool_web_search(query: str):
    """ Search web.

    Args:
        query: search query
    :return: result of WEB search
    """
    search = DuckDuckGoSearchRun()
    # Invoke chain
    result = search.invoke(query)

    return result


def st_app(agent):
    """ Simple search with Ollama and Llama 3.2

    :return: None
    """
    st.set_page_config(page_title="Search WEB")
    st.subheader('Search WEB')

    source_text = st.text_area("Source Text", label_visibility="collapsed", height=270)

    # If the 'Summarize' button is clicked
    if st.button("Search"):
        # Validate inputs
        if not source_text.strip():
            st.error(f"Please provide the missing fields.")
        else:
            try:
                with st.spinner('Please wait...'):
                    # Invoke agent
                    result = agent.invoke({"input": source_text})
                    # Display a success message.
                    st.success(result["output"])
            except Exception as e:
                st.exception(f"An error occurred: {e}")


if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2:latest"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    # register tools
    tools = [tool_web_search]
    # Create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Use the tool_web_search function to answer the user's question."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    # Create agent
    agent = create_tool_calling_agent(llm_ollama, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)


    st_app(agent=agent_executor)
