from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages

if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    messages = [
        SystemMessage("you're a good assistant."),
        HumanMessage("tell me about torch.nn.Module"),
        AIMessage('Base class for all neural network modules.'),
        HumanMessage("What is a state_dict?"),
        AIMessage("A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor."),
        HumanMessage("What is a named_modules?"),
    ]

    result = trim_messages(
        messages,
        # Keep the last <= n_count tokens of the messages.
        strategy="last",
        # highlight-start
        # Remember to adjust based on your model
        # or else pass a custom token_encoder
        token_counter=llm_ollama,
        # highlight-end
        # Most chat models expect that chat history starts with either:
        # (1) a HumanMessage or
        # (2) a SystemMessage followed by a HumanMessage
        # highlight-start
        # Remember to adjust based on the desired conversation
        # length
        max_tokens=45,
        # highlight-end
        # Most chat models expect that chat history starts with either:
        # (1) a HumanMessage or
        # (2) a SystemMessage followed by a HumanMessage
        start_on="human",
        # Most chat models expect that chat history ends with either:
        # (1) a HumanMessage or
        # (2) a ToolMessage
        end_on=("human", "tool"),
        # Usually, we want to keep the SystemMessage
        # if it's present in the original history.
        # The SystemMessage has special instructions for the model.
        include_system=True,
        allow_partial=False,
    )
    print(f"Trim with token counter -----------------------------------------------------------------\n{result}\n")
    result = trim_messages(
        messages,
        # Keep the last <= n_count tokens of the messages.
        strategy="last",
        # highlight-start
        # Remember to adjust based on your model
        # or else pass a custom token_encoder
        token_counter=len,
        # highlight-end
        # Most chat models expect that chat history starts with either:
        # (1) a HumanMessage or
        # (2) a SystemMessage followed by a HumanMessage
        # highlight-start
        # Remember to adjust based on the desired conversation
        # length
        max_tokens=4,
        # highlight-end
        # Most chat models expect that chat history starts with either:
        # (1) a HumanMessage or
        # (2) a SystemMessage followed by a HumanMessage
        start_on="human",
        # Most chat models expect that chat history ends with either:
        # (1) a HumanMessage or
        # (2) a ToolMessage
        end_on=("human", "tool"),
        # Usually, we want to keep the SystemMessage
        # if it's present in the original history.
        # The SystemMessage has special instructions for the model.
        include_system=True,
        allow_partial=False,
    )
    print(f"Trim with message count -----------------------------------------------------------------\n{result}\n")