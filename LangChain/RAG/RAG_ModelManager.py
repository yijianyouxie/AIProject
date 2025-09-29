from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_openai import ChatOpenAI

def setup_language_model(model_config):
    """设置语言模型
    参数：
        model_config 模型配置的字典
            - model_type: local or remote
            - model_path: local模式下的模型路径
            - api_base: API基础URL
            - api_key: API秘钥
            - model_name: remote模型下的模型名称
    返回：
        llm
    """

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
def create_qa_chain(llm, vectorstore):
    """创建问答链
    参数：
        llm 语言模型
        vectorstore 向量存储对象

    返回：
        qa_chain 问答链对象
    """
    # 创建提示模版
    strict_prompt_template = """你是一个严格的文档问答系统。请遵守以下规则：

                                规则：
                                1. 你只能使用下面提供的上下文信息来回答问题
                                2. 绝对禁止使用任何外部知识、常识或个人观点
                                3. 如果上下文没有提供答案，必须回答："文档中未包含此信息"
                                4. 不要解释，不要添加额外信息
                                5. 回答要简洁

                                上下文信息：
                                {context}

                                问题：{question}

                                请仔细检查上下文是否包含问题答案。如果包含，请直接引用上下文内容回答；如果不包含，请明确说明"文档中未包含此信息"。

                                基于文档的回答："""
    strict_prompt_template2 = """<|im_start|>system
                                你是一个严格的文档问答系统。请遵守以下规则：

                                规则：
                                1. 你只能使用下面提供的上下文信息来回答问题
                                2. 绝对禁止使用任何外部知识、常识或个人观点
                                3. 如果上下文没有提供答案，必须回答："文档中未包含此信息"
                                4. 不要解释，不要添加额外信息
                                5. 回答要简洁<|im_end|>
                                <|im_start|>user
                                上下文信息：
                                {context}

                                问题：{question}

                                请仔细检查上下文是否包含问题答案。如果包含，请直接引用上下文内容回答；如果不包含，请明确说明"文档中未包含此信息"。
                                <|im_end|>
                                <|im_start|>assistant
                                基于文档的回答："""
    PROMPT = PromptTemplate(
        template=strict_prompt_template2,
        input_variables=["context","question"]
    )
    # 创建检索
    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs={"k":5}
        # search_type="similarity_score_threshold",  # 使用带阈值的相似度搜索
        # search_kwargs={
        #     "k": 5,                    # 返回最多5个结果
        #     "score_threshold": 0.1     # 只返回相似度高于0.7的文档
        # }
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever = retriever,
        chain_type_kwargs={"prompt":PROMPT},
        return_source_documents =True
    )
    return qa_chain
