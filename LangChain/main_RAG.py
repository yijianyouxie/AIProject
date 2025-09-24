from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.llms import huggingface_pipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import re

def setup_qa_system(file_path, model_path, embedding_model_path):
    """设置基于本地pytorch的知识问答系统"""
    # 1，加载文档
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt') or file_path.endswith('.md') :
        try:
            # 尝试UTF-8编码
            loader = TextLoader(file_path, encoding='utf-8')
        except:
            try:
                # 如果UTF-8失败，尝试GBK编码（常见于中文Windows系统）
                loader = TextLoader(file_path, encoding='gbk')
            except Exception as e:
                print(f"无法加载文本文件: {e}")
                # 可以在这里添加更多编码尝试
                # raise
    else:
        raise ValueError("不支持的文本格式，请使用PDF或TXT格式的文本.")
    print("====正在加载文档。")
    documents = loader.load()

    # 2，分割文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 600,
        chunk_overlap = 100,
        length_function=len,
        separators=["\n# ", "\n## ", "\n### ", "\n#### ", "\n- ", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(f"====文档已分割为{len(texts)}个文本块。")
    print(f"===={texts}")
    # 将文本块写入文件进行调试
    def debug_write_chunks_to_file(texts, filename="text_chunks_debug.txt"):
        """将文本块写入文件，避免控制台显示问题"""
        with open(filename, 'w', encoding='utf-8') as f:
            for i, text in enumerate(texts):
                f.write(f"=== 文本块 {i+1} ===\n")
                f.write(text.page_content)
                f.write("\n" + "="*50 + "\n\n")
        print(f"文本块已写入文件: {filename}")
    debug_write_chunks_to_file(texts, "text_chunks_debug.txt")

    # 3，创建向量存储
    print("====创建本地嵌入模型。")
    embeddings = HuggingFaceEmbeddings(
        model_name = embedding_model_path,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings':False}
    )
    print(f"====embeddings:{embeddings}")

    # 4，创建向量存储
    print("====正在创建向量存储。")
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./local_chroma_db"
    )
    # 测试检索功能
    def test_retrieval(vectorstore, question):
        print(f"\n====测试检索: '{question}' ====")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)
        for i, doc in enumerate(docs):
            print(f"结果 {i+1}: {doc.page_content[:300]}...")
        return docs
    
    test_questions = ["近期重要事件", "C轮融资", "云计算平台"]
    for q in test_questions:
        test_retrieval(vectorstore, q)

    # 5,加载本地pytorch语言模型到GPU
    print("====正在加载语言模型。")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU可用：{torch.cuda.get_device_name(0)}" if device == "cuda" else "GPU不可用，将使用CPU")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype = torch.float16,
        device_map = device,
    )
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        dtype = torch.float16,
        device_map=device,
        max_new_tokens=512,# 最大生成token数
        temperature=0.1,# 较低的温度使输出更确定性
        top_p = 0.9,# 核采样参数
        repetition_penalty=1.3,
        do_sample=False,# False:贪婪解码，更确定但可能更保守;True:随机采样，更有创造性但可能偏离主题
        return_full_text=False,
        # eos_token_id=tokenizer.eos_token_id,
        # pad_token_id=tokenizer.eos_token_id
    )
    llm = HuggingFacePipeline(pipeline = text_generation_pipeline)
    
    # 6,创建适合qwen模型的提示模版
    prompt_template = """请严格基于以下提供的上下文信息回答问题。不要添加任何外部知识。

        上下文信息：
        {context}

        问题：{question}

        请只基于上述上下文回答。如果上下文包含答案，请直接给出答案；如果上下文不包含相关信息，请说"根据上下文无法回答"。

        基于上下文的答案："""
    # 使用最严格的提示模板
    strict_prompt_template = """你是一个严格的文档问答系统。请遵守以下规则：

规则：
1. 你只能使用下面提供的上下文信息来回答问题
2. 绝对禁止使用任何外部知识、常识或个人观点
3. 如果上下文没有提供答案，必须回答："文档中未包含此信息"
4. 不要解释，不要添加额外信息

上下文信息：
{context}

问题：{question}

请仔细检查上下文是否包含问题答案。如果包含，请直接引用上下文内容回答；如果不包含，请明确说明"文档中未包含此信息"。

基于文档的回答："""
    PROMPT = PromptTemplate(
        template=strict_prompt_template,
        input_variables=["context", "question"]
    )

    # 7,创建检索型问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="mmr",# 使用最大边际相关性，提高多样性
            search_kwargs={
                "k":5,# 增加检索数量
                "fetch_k":10,# 增加候选集大小
                "lambda_mult":0.7# 平衡相关性和多样性
                }), # 减少检索数量
        chain_type_kwargs={"prompt":PROMPT},
        return_source_documents = True
    )

    return qa_chain

def ask_question(qa_system, question):
    """提问函数"""
    print(f"\n提问问题：{question}")
    print(f"正在思考")
    try:
        # 先单独测试检索器
        retriever = qa_system.retriever
        retrieved_docs = retriever.get_relevant_documents(question)
        
        print("====检索到的文档（检索器测试）====")
        for i, doc in enumerate(retrieved_docs):
            print(f"文档 {i+1} (相似度: 未知):")
            print(f"内容: {doc.page_content[:300]}...")
            print("---")

        result = qa_system({"query":question})
        answer = result["result"]
        source_docs = result["source_documents"]

        print(f"=====================答案=====================\n：{answer}")
        # 显示来源文档
        if source_docs:
            print("\n====打印答案来源")
            for i, doc in enumerate(source_docs):
                source = doc.metadata.get('source', '未知文档')
                page = doc.metadata.get('page', '未知页码')
                print(f"{i + 1}. {source} 第{page}页")
                # 显示部分预览内容
                content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                print(f"    内容：{content_preview}")
    except Exception as e:
        print(f"提问问题时出错：{e}")

def main():
    """主函数"""
    # 设置路径
    file_path = r"G:\AI\AIProject\LangChain\testLangchain.md"
    model_path = r"G:\AIModels\modelscope_cache\models\Qwen\Qwen1___5-1___8B-Chat"
    embedding_model_path = r"G:\AIModels\modelscope_cache\models\sentence-transformers\all-MiniLM-L6-v2"
    try:
        print("====正在初始化文档问答系统")
        qa_system = setup_qa_system(file_path, model_path, embedding_model_path)
        print("====文档问答系统初始化完成")
        example_questions = [
            "文档的主要主题是什么？",
            "总结下文档的核心内容",
            "文档中提到了哪些重要概念？"
        ]
        print("\n你可以尝试以下问题：")
        for i, q in enumerate(example_questions):
            print(f"{i}. {q}")

        print("\n" + "="*50)
        print("文档问答系统已经就绪。输入'退出'、'quit'或'exit'结束对话")
        print("="*50)
        while True:
            question = input("\n请输入您的问题：").strip()
            if question.lower() in ['quit','退出','exit']:
                print("感谢使用，再见！")
                break
            if question:
                ask_question(qa_system, question)
    except Exception as e:
        print(f"系统初始化失败：{e}")
        print("请检查：1)文件路径，模型路径是否正确；2)GPU空间是否足够")

if __name__ == "__main__":
    main()