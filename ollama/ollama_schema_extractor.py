from langchain_ollama import ChatOllama
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field


class Person(BaseModel):
    """Information about a person."""

    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(default=None, description="The color of the person's hair if known")
    eyes_color: Optional[str] = Field(default=None, description="The color of the person's eyes if known")
    height_in_meters: Optional[float] = Field(default=None, description="Height measured in meters")


if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature).with_structured_output(schema=Person)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            ("human", "{text}"),
        ]
    )
    text = """At 1.82 meters, Elara is a tall girl with long, vibrant auburn hair and striking emerald green eyes, casually dressed in jeans and a shirt."""
    prompt = prompt_template.invoke({"text": text})
    response = llm_ollama.invoke(prompt)

    print(response)
