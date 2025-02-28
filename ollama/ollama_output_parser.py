from langchain.chains.question_answering.map_reduce_prompt import messages
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import pprint

def prompt_template_invoke(llm, prompt_template) -> str:
    """
    Invoke with prompt template

    :param llm: ChatOllama Model
    :param prompt_template: prompt template
    :return ai_msg: LLM response
    """

    ai_msg = llm.invoke(prompt_template)
    return ai_msg.content


if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    # Invoke with output parser
    # Create text
    text_sgd = """
    class torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False, foreach=None, differentiable=False, fused=None)
    
    Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from On the importance of initialization and momentum in deep learning.

    Parameters:
    params (iterable) – iterable of parameters or named_parameters to optimize or iterable of dicts defining parameter groups. When using named_parameters, all parameters in all groups should be named
    lr (float, Tensor, optional) – learning rate (default: 1e-3)
    momentum (float, optional) – momentum factor (default: 0)
    dampening (float, optional) – dampening for momentum (default: 0)
    weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
    nesterov (bool, optional) – enables Nesterov momentum. Only applicable when momentum is non-zero. (default: False)
    maximize (bool, optional) – maximize the objective with respect to the params, instead of minimizing (default: False)
    foreach (bool, optional) – whether foreach implementation of optimizer is used. If unspecified by the user (so foreach is None), we will try to use foreach over the for-loop implementation on CUDA, since it is usually significantly more performant. Note that the foreach implementation uses ~ sizeof(params) more peak memory than the for-loop version due to the intermediates being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer parameters through the optimizer at a time or switch this flag to False (default: None)
    differentiable (bool, optional) – whether autograd should occur through the optimizer step in training. Otherwise, the step() function runs in a torch.no_grad() context. Setting to True can impair performance, so leave it False if you don’t intend to run autograd through this instance (default: False)
    fused (bool, optional) – whether the fused implementation is used. Currently, torch.float64, torch.float32, torch.float16, and torch.bfloat16 are supported. (default: None)
    """
    # Create schemas
    class_name_schema = ResponseSchema(name="class_name",
                                       description="Describe class name")
    class_info_schema = ResponseSchema(name="class_info",
                                       description="Explain what the class does.")
    class_parameters_schema = ResponseSchema(name="class_parameters",
                                             description="Store class parameters as Python dictionarie")
    # Declare response schema
    response_schemas = [class_name_schema, class_info_schema, class_parameters_schema]
    # Create parser
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    # Get format instructions for the output parser
    instructions = parser.get_format_instructions()
    # Create template
    template = """
    From text extract information:

    class_name: describe Pytorch class name
    class_info: describe Pytorch class functionality
    class_parameters: describe Pytorch class parameters
    
    test: {text}
    
    {instructions}
    """
    # Create a chat prompt template from a template string
    chat_template = ChatPromptTemplate.from_template(template=template)
    messages = chat_template.format_messages(text=text_sgd,
                                             instructions=instructions)

    # Invoke
    msg_0 = prompt_template_invoke(llm=llm_ollama, prompt_template=messages)
    # Parse a single string model output
    result_dict = parser.parse(msg_0)

    print(f"Invoke with ResponseSchema --------------------------------------------------------------------\n{msg_0}\n")
    print(f"Parse response with OutputParser --------------------------------------------------------------\n")
    pprint.pp(result_dict)
    print(f"Access the dictionary keys --------------------------------------------------------------------\n")
    print("class_name: ", result_dict["class_name"])
    print("parameter nesterov: ", result_dict["class_parameters"]["nesterov"])