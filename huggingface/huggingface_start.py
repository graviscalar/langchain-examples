from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
import os


def model_load_save(model_name: str, dir_save: str):
    """ Download and save hugginface model files

    :param model_name: name of model on hugginface website
    :param dir_save: the local directory to save model files
    :return: result of directory creation
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # if not os.path.exists(dir_save):
    #     os.makedirs(dir_save)
    tokenizer.save_pretrained(dir_save)
    model.save_pretrained(dir_save)
    return os.path.exists(dir_save)


if __name__ == '__main__':
    model_id = "microsoft/phi-2"
    dir_model = "models/hf-frompretrained/microsoft/phi-2/"
    # Download and save model. RUN THIS FUNCTION ONLY 1 TIME. Model will be saved to your local pc.
    model_load_save(model_name=model_id, dir_save=dir_model) # comment this function after saving model to local pc.

    tokenizer = AutoTokenizer.from_pretrained(dir_model)
    model = AutoModelForCausalLM.from_pretrained(dir_model)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    # HuggingFace Pipeline API
    hf = HuggingFacePipeline(pipeline=pipe)

    template = """Question: {question}."""
    prompt = PromptTemplate.from_template(template)
    # Create chain
    chain = prompt | hf

    question = "What is nn.Conv2d?"
    # Chain invoke
    print(chain.invoke({"question": question}))