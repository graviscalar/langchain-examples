from langchain_ollama import ChatOllama
from typing import Optional
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict


# Pydantic
class PytorchClassDescriptionPydantic(BaseModel):
    """ Description of the Pytorch class."""

    class_name: str = Field(description="PyTorch class name")
    class_description: str = Field(description="Description of the function")
    class_params: Optional[int] = Field(default=None, description="How many parameters does the function have? Answer as number")


# TypedDict
class PytorchClassDescriptionTypedDict(TypedDict):
    """ Description of the Pytorch class."""

    class_name: Annotated[str, ..., "PyTorch class name"]
    class_description: Annotated[str, ..., "Description of the function"]
    class_params: Annotated[Optional[int], None, "How many parameters does the function have?  Answer as number"]


if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    # Pydantic class
    structured_llm = llm_ollama.with_structured_output(PytorchClassDescriptionPydantic)

    result = structured_llm.invoke("Tell me about torch.nn.Conv1d")
    print(f"Pydantic output ------------------------------------------------------------------------------\n{result}\n")

    # TypedDict class
    structured_llm = llm_ollama.with_structured_output(PytorchClassDescriptionTypedDict)

    result = structured_llm.invoke("Tell me about torch.nn.Conv1d")
    print(f"TypedDict output -----------------------------------------------------------------------------\n{result}\n")

    # JSON schema
    json_schema = {
        "title": "Pytorch_class",
        "description": " Description of the Pytorch class.",
        "type": "object",
        "properties": {
            "class_name": {
                "type": "string",
                "description": "PyTorch class name",
            },
            "class_description": {
                "type": "string",
                "description": "Description of the function",
            },
            "class_params": {
                "type": "integer",
                "description": "How many parameters does the function have?  Answer as number",
                "default": None,
            },
        },
        "required": ["class_name", "description"],
    }
    structured_llm = llm_ollama.with_structured_output(json_schema)

    result = structured_llm.invoke("Tell me about torch.nn.Conv1d")
    print(f"JSON output ----------------------------------------------------------------------------------\n{result}\n")
