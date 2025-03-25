from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import duckdb


def get_var_name(var, scope):
    a = [name for name, value in scope.items() if value is var]
    return a[0]


if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    # Create database
    db = duckdb.read_csv("../data/docs/pytorch_containers.csv")

    # Create schema
    schema = duckdb.sql(f"DESC {get_var_name(db, locals())}")

    # Create prompt
    sql_prompt = PromptTemplate.from_template("""
    Given the following table schema for the table `db`:

    {schema}

    Write an SQL statement to answer the following question:

    {question}

    Provide all rows form the table - eg select * from.
    Provide a limit where applicable.
    Respond with only the SQL statement.
    """)

    # Create chain
    sql_chain = sql_prompt | llm_ollama | StrOutputParser()

    # Invoke chain
    sql_query = sql_chain.invoke({
        "schema": schema,
        "question": "Is there a row with a description in the explanation of the Base class?"
    })

    print(f"Invoke sql chain --------------------------------------------------------------------\n{sql_query}\n")
