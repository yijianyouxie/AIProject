import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# 设置API秘钥
os.environ["OPENAI_API_KEY"] = ""

# 初始化模型
llm = ChatOpenAI(
    model_name = "gpt-4o",
    temperature=0.7
)

# 定义模版
template = "请用{language}翻译这句话：{text}"
prompt = PromptTemplate(
    input_variables=["language", "text"],
    template = template
)
formated_prompt = prompt.format(language="法语", text="我爱编程")
print(formated_prompt)

reponse = llm.invoke(formated_prompt)
print(reponse)