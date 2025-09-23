import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"GPU可用：{torch.cuda.get_device_name(0)}" if device == "cuda" else "GPU不可用，将使用CPU")

# 加载开源模型
model_path = "G:\AIModels\modelscope_cache\models\Qwen\Qwen1___5-1___8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    dtype=torch.float16,
)

# 创建文本生成管道
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    # top_p=0.9,
    # repetition_penalty=1.1,
)

# 用LangChain封装模型
llm = HuggingFacePipeline(pipeline=pipe)

# 为Qwen模型设计专门的提示词格式
def format_qwen_prompt(instruction, input_text=None, type = 0):
    """为Qwen模型格式化提示词"""
    if type == 0:
        return f"<|im_start|>system\n你是一个有帮助的助手，专门用来生成文章标题。<|im_end|>\n<|im_start|>user\n{instruction}: {input_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        return f"<|im_start|>system\n你是一个有帮助的助手，专门根据文章标题生成文章。<|im_end|>\n<|im_start|>user\n{instruction}: {input_text}<|im_end|>\n<|im_start|>assistant\n"
    # if input_text:
    #     return f"<|im_start|>system\n你是一个有帮助的助手，专门根据文章标题生成文章。type = {type}<|im_end|>\n<|im_start|>user\n{instruction}: {input_text}<|im_end|>\n<|im_start|>assistant\n"
    # else:
    #     return f"<|im_start|>system\n你是一个有帮助的助手，专门用来生成文章标题。<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

# 清理输出的函数
def clean_output(output):
    # 移除Qwen特定的标记
    cleaned = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', output, flags=re.DOTALL)
    cleaned = re.sub(r'<\|im_start\|>assistant', '', cleaned)
    # 移除多余的空行和空格
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # 移除可能的多余引号
    cleaned = cleaned.replace('"', '').replace("'", "")
    return cleaned
    return output

# 创建标题生成链
def create_title_prompt(inputs):
    theme = inputs["theme"]
    instruction = "请根据给定主题生成一个简洁的文章标题，只输出标题不要其他内容"
    # print(f"****theme={theme}****")
    prompt_text = format_qwen_prompt(instruction, theme, 0)
    return prompt_text

title_prompt_chain = RunnableLambda(create_title_prompt)
title_chain = title_prompt_chain | llm | StrOutputParser()

# 创建文章生成链
def create_article_prompt(inputs):
    title = inputs["title"]
    instruction = f"请根据给定标题撰写一篇完整的文章，内容详实结构清晰，字数在300字以内"
    # print(f"****title={title}****")
    prompt_text = format_qwen_prompt(instruction, title, 1)
    return prompt_text

article_prompt_chain = RunnableLambda(create_article_prompt)
article_chain = article_prompt_chain | llm | StrOutputParser()

# 清理标题输出的函数
def clean_title_output(output):
    # print("=====output=" + output)
    cleaned = clean_output(output)
    # 如果标题仍然包含多余内容，尝试提取标题部分
    if ":" in cleaned:
        # 尝试提取冒号后的内容
        parts = cleaned.split(":")
        if len(parts) > 1:
            cleaned = parts[-1].strip()
    return {"title": cleaned}

# 创建组合链 - 使用RunnablePassthrough传递输入
overall_chain = (
    RunnablePassthrough.assign(title=title_chain)
    | RunnableLambda(lambda x: clean_title_output(x["title"]))
    | RunnablePassthrough.assign(article=article_chain)
)

# 执行链
theme = "Unity引擎"

# 获取结果
result = overall_chain.invoke({"theme": theme})

print("====标题:" + result["title"])
print("====文章:" + clean_output(result["article"]))