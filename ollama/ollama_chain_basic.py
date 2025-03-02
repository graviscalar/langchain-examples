from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

if __name__ == "__main__":
    # use your Ollama model. Verify loaded models with 'ollama list'
    model = "llama3.2"
    temperature = 0

    llm_ollama = ChatOllama(model=model,
                            temperature=temperature)

    # Create text
    txt_conv1d = """
    class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    
    Applies a 1D convolution over an input signal composed of several input planes.
    
    Parameters:
        in_channels (int) – Number of channels in the input image
        out_channels (int) – Number of channels produced by the convolution
        kernel_size (int or tuple) – Size of the convolving kernel
        stride (int or tuple, optional) – Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) – Padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
        groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
        padding_mode (str, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    """
    # Create template
    template = """Extract the class name and the number of parameters from the description in {text}."""
    # Create a chat prompt template from a template string
    chat_template = ChatPromptTemplate.from_template(template)
    chain = chat_template | llm_ollama
    msg_txt = chain.invoke({"text": txt_conv1d})
    print(f"Invoke simple chain -----------------------------------------------------------------\n{msg_txt.content}\n")

    # Invoke with StrOutputParser
    chain = chat_template | llm_ollama | StrOutputParser()
    msg_txt = chain.invoke({"text": txt_conv1d})
    print(f"Invoke simple chain with StrOutputParser ----------------------------------------------------\n{msg_txt}\n")
