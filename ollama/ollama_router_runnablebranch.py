from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch

if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "gemma3:4b"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    # Setup chain for topic
    topic_chain = (
            PromptTemplate.from_template(
            """Given the user question below, classify it as either being about `Python`, `Pytorch`, or `Scikit-learn`.
          
              Do not respond with more than one word.
          
              <question>
              {question}
              </question>
          
              Classification:"""
            )
            | llm_ollama
            | StrOutputParser()
    )

    # Setup sub-chains

    # Python chain
    python_chain = PromptTemplate.from_template(
        """You are an expert in Python. \
    Always answer questions starting with "As Python expert". \
    Respond to the following question:

    Question: {question}
    Answer:"""
    ) | llm_ollama

    # Pytorch chain
    pytorch_chain = PromptTemplate.from_template(
        """You are an expert in Pytorch. \
    Always answer questions starting with "As Pytorch expert". \
    Respond to the following question:

    Question: {question}
    Answer:"""
    ) | llm_ollama

    # Scikit-learn chain
    scikit_chain = PromptTemplate.from_template(
        """You are an expert in Scikit-learn. \
    Always answer questions starting with "As Scikit-learn expert". \
    Respond to the following question:

    Question: {question}
    Answer:"""
    ) | llm_ollama

    # Create branch
    branch = RunnableBranch(
        (lambda x: "python" in x["topic"].lower(), python_chain),
        (lambda x: "pytorch" in x["topic"].lower(), pytorch_chain),
        (lambda x: "scikit-learn" in x["topic"].lower(), scikit_chain),
        scikit_chain,
    )

    # Create a final chain
    final_chain = {"topic": topic_chain, "question": lambda x: x["question"]} | branch

    # Route to Python chain
    result = final_chain.invoke({"question": "tell me about decorators"})
    print(f"Routing to Python chain ----------------------------------------------------------------------\n{result}\n")

    # Route to Pytorch chain
    result = final_chain.invoke({"question": "tell me about nn.Module"})
    print(f"Routing to Pytorch chain ---------------------------------------------------------------------\n{result}\n")

    # Route to Scikit-learn chain
    result = final_chain.invoke({"question": "tell me about random forest"})
    print(f"Routing to Scikit-learn chain ----------------------------------------------------------------\n{result}\n")
