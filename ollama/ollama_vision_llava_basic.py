from langchain_ollama import ChatOllama
import base64
from io import BytesIO
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser


def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def image_prompt_to_langchain_prompt(data):
    content = []

    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    text_part = {"type": "text", "text": text}

    content.append(image_part)
    content.append(text_part)

    return [HumanMessage(content=content)]


if __name__ == "__main__":
    # use your Llava model. Verify loaded models with 'ollama list'
    # ollama.pull("llava")  # you can pull model from python
    model = "llava"  # more info here https://ollama.com/library/llava
    temperature = 0
    llm_llava = ChatOllama(model=model,
                           temperature=temperature)
    # Load your image
    image_path = "../data/img/road_0.jpg"
    dt_image = Image.open(image_path)
    # Convert image to the base64
    image_b64 = image_to_base64(dt_image)
    # Setup chain
    chain = image_prompt_to_langchain_prompt | llm_llava | StrOutputParser()
    # Invoke chain
    query_chain = chain.invoke({"text": "Describe the content of the image.", "image": image_b64})
    print(query_chain)
