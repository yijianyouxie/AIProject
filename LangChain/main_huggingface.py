import re
import uuid
import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
# LLMChain
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import initialize_agent, AgentType, Tool, AgentExecutor, create_react_agent
# from langgraph.prebuilt import create_react_agent
from langchain_huggingface import ChatHuggingFace
import json

# region ====公共部分====
device = "cuda"
# 检查GPU是否可用
if torch.cuda.is_available():
    print(f"GPU可用：{torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("GPU不可用，将使用CPU")
    device = "cpu"


# 加载开源模型
model_path = "G:\AIModels\modelscope_cache\models\Qwen\Qwen1___5-1___8B-Chat"
# model_path = "G:\AIModels\modelscope_cache\models\Qwen\Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    dtype=torch.float16,
    # load_in_4bit=False,
)
# 关键修正1：获取Qwen1.5的特殊Token ID（用convert_tokens_to_ids替代get_command_id）
im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
eos_id = tokenizer.eos_token_id  # 标准终止符ID
# print(f"===========im_start_id: {im_start_id}, im_end_id: {im_end_id}, eos_id: {eos_id}")
# 创建文本生成管道
pipe = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 512,
    temperature = 0.3,
    do_sample=True,
    # device = 0 if device == "cuda" else -1
    return_full_text=False,
    pad_token_id=tokenizer.pad_token_id,
    # Qwen1.5终止符：生成到<|im_end|>或下一个<|im_start|>自动停止
    eos_token_id=[ im_end_id, eos_id, im_start_id],  # 模型生成到这些ID时停止
)

# 用LangChain封装模型
llm = HuggingFacePipeline(pipeline = pipe)
# endregion

# region ====翻译任务 例子====
# template = "请用{language}翻译这句话：{text}\n翻译结果"
# prompt = PromptTemplate(
#     input_variables=["language", "text"],
#     template=template
# )
# formatted_prompt = prompt.format(language = "英语", text = "我爱编程")

# # 生成结果
# response = llm.invoke(formatted_prompt)
# print("======" + response)
# endregion

# region ====LLMChain 例子====
# prompt = PromptTemplate(
#     input_variables=["topic"],
#     template="请写一个关于{topic}的3句话简介。"
# )
# chain=LLMChain(llm = llm, prompt = prompt)
# result = chain.invoke(topic = "人工智能")
# print("====" + result)
# endregion

# region ====多链条 例子====
# 为Qwen模型设计专门的提示词格式
# def format_qwen_prompt(instruction, input_text=None, type = 0):
#     """为Qwen模型格式化提示词"""
#     if type == 0:
#         return f"<|im_start|>system\n你是一个有帮助的助手，专门用来生成文章标题。<|im_end|>\n<|im_start|>user\n{instruction}: {input_text}<|im_end|>\n<|im_start|>assistant\n"
#     else:
#         return f"<|im_start|>system\n你是一个有帮助的助手，专门根据文章标题生成文章。<|im_end|>\n<|im_start|>user\n{instruction}: {input_text}<|im_end|>\n<|im_start|>assistant\n"
# # 清理输出的函数
# def clean_output(output):
#     # 移除Qwen特定的标记
#     cleaned = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', output, flags=re.DOTALL)
#     cleaned = re.sub(r'<\|im_start\|>assistant', '', cleaned)
#     # 移除多余的空行和空格
#     cleaned = re.sub(r'\s+', ' ', cleaned).strip()
#     # 移除可能的多余引号
#     cleaned = cleaned.replace('"', '').replace("'", "")
#     return cleaned
# # 创建标题生成链
# def create_title_prompt(inputs):
#     theme = inputs["theme"]
#     instruction = "请根据给定主题生成一个简洁的文章标题，只输出标题不要其他内容"
#     prompt_text = format_qwen_prompt(instruction, theme, 0)
#     return prompt_text
# title_prompt_chain = RunnableLambda(create_title_prompt)
# title_chain = title_prompt_chain | llm | StrOutputParser()
# # 创建文章生成链
# def create_article_prompt(inputs):
#     title = inputs["title"]
#     instruction = f"请根据给定标题撰写一篇完整的文章，内容详实结构清晰，字数在300字以内"
#     prompt_text = format_qwen_prompt(instruction, title, 1)
#     return prompt_text
# article_prompt_chain = RunnableLambda(create_article_prompt)
# article_chain = article_prompt_chain | llm | StrOutputParser()
# # 清理标题输出的函数
# def clean_title_output(output):
#     cleaned = clean_output(output)
#     # 如果标题仍然包含多余内容，尝试提取标题部分
#     if ":" in cleaned:
#         # 尝试提取冒号后的内容
#         parts = cleaned.split(":")
#         if len(parts) > 1:
#             cleaned = parts[-1].strip()
#     return {"title": cleaned}
# # 创建组合链 - 使用RunnablePassthrough传递输入
# overall_chain = (
#     RunnablePassthrough.assign(title=title_chain)
#     | RunnableLambda(lambda x: clean_title_output(x["title"]))
#     | RunnablePassthrough.assign(article=article_chain)
# )
# # 执行链
# theme = "Unity引擎"
# # 获取结果
# result = overall_chain.invoke({"theme": theme})
# print("====标题:" + result["title"])
# print("====文章:" + clean_output(result["article"]))
# endregion

# region ====Memory 例子====
# # 创建存储回话历史的字典
# store = {}
# # 获取或创建回话历史的函数
# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         history = ChatMessageHistory()
#         # 关键修正1：用SystemMessage类（BaseMessage子类）添加系统提示，而非字典
#         system_prompt = """你是简洁的对话助手，严格遵守：
#                         1. 只回答用户当前问题，不添加无关内容；
#                         2. 回答不超过2句话。
#                         3. 无论什么情况，回答末尾必须加`<|im_end|>`标签。"""
#         history.add_message(SystemMessage(content=system_prompt))  # 正确：SystemMessage实例
#         store[session_id] = history
#     return store[session_id]
# # 格式化对话内容
# def format_qwen1_5_history(history: ChatMessageHistory) -> str:
#     formatted = []
#     # 关键修正2：遍历的是BaseMessage实例，有type属性
#     # 提取系统提示（SystemMessage的type是"system"）
#     system_content = next(
#         (msg.content for msg in history if msg.type == "system"),  # msg是BaseMessage，有type
#         ""
#     )
#     if system_content:
#         formatted.append(f"<|im_start|>system\n{system_content}<|im_end|>")
    
#     # 拼接用户/助手消息（HumanMessage.type是"human"，AIMessage.type是"ai"）
#     for msg in history:
#         if isinstance(msg, HumanMessage):  # 更严谨的判断方式（推荐）
#             formatted.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
#         elif isinstance(msg, AIMessage):
#             formatted.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
    
#     return "\n".join(formatted)

# def format_history_step(input_dic:dict) -> str:
#     """将原历史消息转为为qwen格式"""
#     return {
#         "formatted_history": format_qwen1_5_history(input_dic["history"]),
#         "input":input_dic["input"]
#     }
# # 提示词模版：历史消息 + 当前user输入 + assistant前缀
# qwen_prompt = PromptTemplate(
#     input_variables=["formatted_history", "input"],
#     template="""{formatted_history}
#     <|im_start|>user\n{input}<|im_end|>
#     <|im_start|>assistant
#     """# 末尾不加<|im_end|>，留给模型生成后自动添加
# )
# def clean_qwen_output(output:str) -> str:
#     return output.replace("<|im_end|>", "").strip()
# def print_raw_prompt(prompt: str) -> str:
#         """打印发送给模型的原始提示（未经过任何处理的字符串）"""
#         print("\n" + "="*50)
#         print("【发送给模型的原始提示内容】：")
#         print(prompt)  # 这就是模型实际接收的输入字符串
#         print("="*50 + "\n")
#         return prompt  # 打印后继续传递给下一个步骤

# def print_raw_output(raw_output: str) -> str:
#     """打印模型返回的原始输出（包含<|im_end|>等标签）"""
#     print("\n" + "="*50)
#     print("【模型生成的原始输出内容】：")
#     print(raw_output)  # 这就是模型未经过清理的原始输出
#     print("="*50 + "\n")
#     return raw_output  # 打印后继续传递给清理步骤
# # 组合链条（简洁高效，适配轻量级模型）
# chain = (
#     format_history_step  # 历史格式转换
#     | qwen_prompt        # 拼接Qwen1.5标准提示词
#     # | print_raw_prompt  # 打印原始提示
#     | llm                # 模型生成
#     # | print_raw_output  # 打印原始输出
#     | clean_qwen_output  # 清理输出
#     | StrOutputParser()  # 转为纯文本
# )

# # 3. 包装对话历史（自动管理session_id对应的历史）
# conversation = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key="input",       # 用户输入的键名
#     history_messages_key="history",   # 对话历史的键名
#     output_messages_key="output"      # 模型输出存入历史的键名
# )

# # --------------------------
# # 6. 测试Qwen1.5-1.8B-Chat对话（验证记忆与简洁性）
# # --------------------------
# if __name__ == "__main__":
    # # 生成唯一会话ID（不同会话历史独立）
    # session_id = str(uuid.uuid4())
    # print(f"当前会话ID：{session_id}\n")

    # # 第一次对话：告知名字
    # print("用户：你好，我叫小明。你是谁？")
    # response1 = conversation.invoke(
    #     {"input": "你好，我叫小明。你是谁？"},
    #     config={"configurable": {"session_id": session_id}}
    # )
    # print(f"Qwen1.5-1.8B-Chat：{response1}\n")

    # # 第二次对话：验证名字记忆
    # print("用户：我刚才告诉你我的名字了吗？另外，我的年龄50岁了。")
    # response2 = conversation.invoke(
    #     {"input": "我刚才告诉你我的名字了吗？另外，我的年龄50岁了"},
    #     config={"configurable": {"session_id": session_id}}
    # )
    # print(f"Qwen1.5-1.8B-Chat：{response2}")

    # # 第三次对话：验证名字记忆
    # print("用户：我告诉你我的年龄了吗？")
    # response3 = conversation.invoke(
    #     {"input": "我告诉你我的年龄了吗？"},
    #     config={"configurable": {"session_id": session_id}}
    # )
    # print(f"Qwen1.5-1.8B-Chat：{response3}")
# endregion

# ====使用外部工具 例子====
## 这是传统代理的实现方式
python_tool = PythonREPLTool()
tools = [
    Tool(
        name="PythonREPL",
        func=python_tool.run,
        description="当需要计算数学问题，处理数据时使用"
    )
]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
result = agent.invoke("计算1+2+...+100的和，用Python代码实现。")
print("====结果\n")
print(result['output'])
print(json.dumps(result, indent=2, ensure_ascii=False))