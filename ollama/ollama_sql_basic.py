from langchain_ollama import ChatOllama
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)
    # Open sql database
    db = SQLDatabase.from_uri("sqlite:///../data/db/chinook.db")

    # Get names of tables available
    print(f"Names of tables available -----------------------------------------------\n{db.get_usable_table_names()}\n")

    # Convert question to sql query
    create_query = create_sql_query_chain(llm_ollama, db)

    # Execute SQL query
    execute_query = QuerySQLDatabaseTool(db=db)

    # chain = write_query | execute_query
    combined_chain = create_query | execute_query

    # Create question
    question = "How many customers are there?"

    # run the chain
    result = combined_chain.invoke({"question": question})

    print(f"Invoke chain with SQL query ------------------------------------------------------------------\n{result}\n")
