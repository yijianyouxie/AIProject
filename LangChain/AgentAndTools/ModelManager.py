from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    GenerationConfig
)
import torch
from langchain_openai import ChatOpenAI

model_config1 ={
    "model_type":"local",
    "model_path":r"G:\AIModels\modelscope_cache\models\Qwen\Qwen3-4B-Instruct-2507"
}
# 配置2 使用本地部署ollama
model_config2 ={
    "model_type":"remote",
    "api_base":"http://localhost:11434/v1", # ollama默认地址
    "api_key":"ollama", # ollama不需要真实秘钥，但需要提供一个值
    "model_name":"qwen:7b"
}
# 配置3 使用其他兼容OpenAI API的服务
model_config3 = {
    "model_type":"remote",
    "api_base":"https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key":"sk-76cc591b40314d239cf44a83737a7cc4",
    "model_name":"qwen-plus"
}

def loadModel(number = 0):
    """加载语言模型"""
    if number == 0:
        model_config = model_config1
    elif number == 1:
        model_config = model_config3
    else:
        print(f"模型创建失败。number = {number}")
        return None
    
    model_type = model_config.get("model_type")
    if model_type == "local":
        print("====正在加载本地语言模型====")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"GPU可用：{torch.cuda.get_device_name(0)}" if device == "cuda" else "GPU不可用，将使用CPU.")
        model_path = model_config.get("model_path")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map=device
        )
        text_generation_pipeline = pipeline(
            "text-generation",
            model = model,
            tokenizer = tokenizer,
            dtype = torch.float16,
            device_map = device,
            max_new_tokens = 512,
            temperature = 0.6,
            top_p = 0.9,# 核采样参数 top_p 就像是一个“智能过滤器”，它确保模型只在那些“总体上看起来合理”的选项中进行随机选择，从而巧妙地平衡了生成的准确性和趣味性
            repetition_penalty=1.3,
            do_sample=True,# False:贪婪解码，更确定但可能更保守;True:随机采样，更有创造性但可能偏离主题
            return_full_text=False,
            # return_full_text=True,# 要是改成True会提示模版的内容也输出出来还是保持False
        )
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        print("====本地语言模型加载成功====")

    elif model_type == "remote":
        print("====正在加载远程语言模型====")
        api_base = model_config.get("api_base")
        api_key = model_config.get("api_key")
        model_name = model_config.get("model_name")
        llm = ChatOpenAI(
            model_name= model_name,
            openai_api_base=api_base,
            openai_api_key=api_key,
            temperature=0.1,
            max_tokens = 512
        )
        print("====远程语言模型加载成功====")

    return llm