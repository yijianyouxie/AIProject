import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# 设置API秘钥
os.environ["OPENAI_API_KEY"] = "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"

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