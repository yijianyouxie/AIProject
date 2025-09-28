from RAG_VectorStoreManager import setup_vector_store
from RAG_ModelManager import setup_language_model, create_qa_chain

def ask_question(qa_system, question):
    """提问函数"""
    print(f"\n提问问题：{question}")
    print(f"正在思考")
    try:
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
    input_path = r"G:\AI\AIProject\LangChain\documents_RAG"  # 可以指向文件夹或单个文件
    # model_path = r"G:\AIModels\modelscope_cache\models\Qwen\Qwen1___5-1___8B-Chat"
    model_path = r"G:\AIModels\modelscope_cache\models\Qwen\Qwen3-4B-Instruct-2507"
    embedding_model_path = r"G:\AIModels\modelscope_cache\models\sentence-transformers\all-MiniLM-L6-v2"
    vectorstore_path = r"./local_chroma_db"  # 向量存储路径
    try:
        print("====正在初始化文档问答系统====")
        vectorstore, _ = setup_vector_store(input_path, embedding_model_path, vectorstore_path)
        llm = setup_language_model(model_path)
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
                ask_question(qa_system, question)
    except Exception as e:
        print(f"系统初始化失败。{e}")
        import traceback
        traceback.print_exc()
        print("请检查：1)文件路径，模型路径是否正确；2)GPU空间是否足够")

if __name__ == "__main__":
    main()