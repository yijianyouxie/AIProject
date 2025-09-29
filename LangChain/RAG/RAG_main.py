from RAG_VectorStoreManager import setup_vector_store, validate_vector_store
from RAG_ModelManager import setup_language_model, create_qa_chain
import sys

def ask_question(qa_system, vectorstore, question):
    """提问函数"""
    print(f"\n提问问题：{question}")
    print(f"正在思考")
    try:
        # 详细检索诊断
        print("====详细检索诊断====")
        
        # 方法1: 直接相似度搜索
        print("1. 直接相似度搜索:")
        docs_with_scores = vectorstore.similarity_search_with_score(question, k=5)
        # 添加分数分析
        print(f"分数范围分析:")
        scores = [score for _, score in docs_with_scores]
        print(f"  最小分数: {min(scores):.4f}")
        print(f"  最大分数: {max(scores):.4f}")
        print(f"  平均分数: {sum(scores)/len(scores):.4f}")
        for i, (doc, score) in enumerate(docs_with_scores):
            has_tech_spec = question in doc.page_content
            print(f"  文档 {i+1} (分数: {score:.4f}, 包含{question}: {has_tech_spec}):")
            print(f"    内容预览: {doc.page_content[:100]}...")
        
        print("\n2. 最大边际相关性(MMR)搜索:")
        # 方法2: MMR搜索
        mmr_docs = vectorstore.max_marginal_relevance_search(question, k=3)
        for i, doc in enumerate(mmr_docs):
            has_tech_spec = question in doc.page_content
            print(f"  文档 {i+1} (包含{question}: {has_tech_spec}):")
            print(f"    内容预览: {doc.page_content[:100]}...")
        
        # 检查嵌入模型
        print(f"\n3. 嵌入模型测试:")
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=r"G:\AIModels\modelscope_cache\models\sentence-transformers\all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'}
        )
        query_embedding = embeddings.embed_query(question)
        print(f"  查询嵌入维度: {len(query_embedding)}")
        print(f"  查询嵌入前5个值: {query_embedding[:5]}")

        result = qa_system.invoke({"query":question})
        answer = result["result"]
        source_docs = result["source_documents"]
        print("="*50 + "答案" + "="*50)
        print(f":{answer}")

        # 创建一个数组，记录已经打印的答案来源
        printedArr = []
        has = False
        if source_docs:
            print("\n====打印答案来源====")
            for i, doc in enumerate(source_docs):
                source = doc.metadata.get('source', '未知文档')
                pageinfo = doc.metadata.get('page', '未知页码')
                if pageinfo != '未知页码':
                    pageinfo = f"第{pageinfo}页"
                
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                # 在数组中查找是否存在
                for item in printedArr:
                    if item == content_preview:
                        has = True
                        break
                if not has:
                    printedArr.append(content_preview)
                    print(f"来源{i+1}. {source} {pageinfo}")
                    print(f"内容" + "=" * 50 + f"\n:{content_preview}")

    except Exception as e:
        print(f"提问问题出错。{e}")
def main():
    """主函数"""
    # 设置路径
    input_path = r"G:\AI\AIProject\LangChain\RAG\documents_RAG"  # 可以指向文件夹或单个文件
    # model_path = r"G:\AIModels\modelscope_cache\models\Qwen\Qwen1___5-1___8B-Chat"
    model_path = r"G:\AIModels\modelscope_cache\models\Qwen\Qwen3-4B-Instruct-2507"
    embedding_model_path = r"G:\AIModels\modelscope_cache\models\sentence-transformers\all-MiniLM-L6-v2"
    vectorstore_path = r"./local_chroma_db"  # 向量存储路径

    # 进行本地和远程配置
    # 配置1 本地模型
    model_config ={
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

    try:
        print("====正在初始化文档问答系统====")
        vectorstore, _ = setup_vector_store(input_path, embedding_model_path, vectorstore_path)

        # 验证向量存储
        test_questions = ["技术规范", "API接口", "数据备份", "公司简介"]
        validate_vector_store(vectorstore, test_questions)

        number = 0
        if len(sys.argv) > 1:
            number = int(sys.argv[1])
        print(f"====模型使用远程模型" if number > 0 else f"====模型使用本地模型")
        llm = setup_language_model(model_config3 if number > 0 else model_config)
        qa_system = create_qa_chain(llm, vectorstore)
        print("====文档问答系统初始化完成====")
        
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
            if question.lower() in ['quit', '退出', 'exit']:
                print("感谢使用，再见！")
                break
            if question:
                ask_question(qa_system, vectorstore, question)
    except Exception as e:
        print(f"系统初始化失败。{e}")
        import traceback
        traceback.print_exc()
        print("请检查：1)文件路径，模型路径是否正确；2)GPU空间是否足够")

if __name__ == "__main__":
    main()