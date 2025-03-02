from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    # Create text
    txt_loss = """
    class torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
    
    This criterion computes the cross entropy loss between input logits and target.

    It is useful when training a classification problem with C classes. If provided, the optional argument weight should be a 1D Tensor assigning weight to each of the classes. This is particularly useful when you have an unbalanced training set.

    Parameters:
        weight (Tensor, optional) – a manual rescaling weight given to each class. If given, has to be a Tensor of size C and floating point dtype
        size_average (bool, optional) – Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True
        ignore_index (int, optional) – Specifies a target value that is ignored and does not contribute to the input gradient. When size_average is True, the loss is averaged over non-ignored targets. Note that ignore_index is only applicable when the target contains class indices.
        reduce (bool, optional) – Deprecated (see reduction). By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average. Default: True
        reduction (str, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the weighted mean of the output is taken, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'
        label_smoothing (float, optional) – A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution as described in Rethinking the Inception Architecture for Computer Vision. Default: 0.0
    
    
    torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    
    Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input x and target y.
    
    Parameters:
        size_average (bool, optional) – Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True
        reduce (bool, optional) – Deprecated (see reduction). By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average. Default: True
        reduction (str, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'
    """

    # Create first template
    template_0 = """From {text}, what is the name of the class and its parameters that are responsible for the {loss}?"""
    # Create a chat prompt template from a template string
    chat_template_0 = ChatPromptTemplate.from_template(template_0)
    # Create first chain
    chain_0 = chat_template_0 | llm_ollama

    # Create second template
    template_1 = """Does class {class} have any deprecated parameters?"""
    # Create a chat prompt template from a template string
    chat_template_1 = ChatPromptTemplate.from_template(template_1)
    # Create second chain
    chain_1 = chat_template_1 | llm_ollama

    # Create sequential chain
    sequential_chain = chain_0 | chain_1

    msg_txt = sequential_chain.invoke({"text": txt_loss, "loss":"mean squared error"})
    print(f"Invoke Sequential chain -------------------------------------------------------------\n{msg_txt.content}\n")


