from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch
import os
import hashlib
import json
from datetime import datetime

def get_file_hash(file_path):
    """计算文件的MD5 hash值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda:f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
def get_folder_hash(folder_path):
    """计算文件夹中所有支持文件的哈希值"""
    supported_extensions = ['.pdf', '.tex', '.md']
    file_hashs = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in supported_extensions):
                file_path = os.path.join(root, file)
                file_hashs.append(get_file_hash(file_path))
    # 对所有文件的哈希值进行排序并计算总体hash
    file_hashs.sort()
    combined_hash = hashlib.md5(''.join(file_hashs).encode()).hexdigest()
    return combined_hash
def get_input_path_hash(input_path):
    """根据输入路径（文件或文件夹）计算哈希值"""
    if os.path.isfile(input_path):
        return get_file_hash(input_path)
    elif os.path.isdir(input_path):
        return get_folder_hash(input_path)
    else:
        raise ValueError(f"路径不存在：{input_path}")
def should_recreate_vectorstore(vectorstore_path, current_hash):
    """检查是否需要重新创建向量存储"""
    metadata_file = os.path.join(vectorstore_path, "metadata.json")
    if not os.path.exists(vectorstore_path):
        return True, "向量存储目录不存在"
    if not os.path.exists(metadata_file):
        return True, "元数据文件不存在"
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            medadata = json.load(f)
        if medadata.get('document_hash') != current_hash:
            return True, f"文档已变更"
        return False, "向量存储有效"
    except Exception as e:
        return True, f"读取元数据失败：{e}"
def save_vectorstore_metadata(vectorstore_path, document_hash, document_paths):
    """保存向量存储的源数据"""
    metadata = {
        'document_hash':document_hash,
        'document_paths':document_paths,
        'created_time':datetime.now().isoformat(),
        'embedding_model':'sentence-transformer/all-MiniLM-L6-v2'
    }
    os.makedirs(vectorstore_path, exist_ok=True)
    metadata_file = os.path.join(vectorstore_path, 'metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
def load_documents_from_folder(folder_path):
    """从文件夹加载所有支持的文档"""
    supported_extensions = {
        '.pdf':PyPDFLoader,
        '.txt':TextLoader,
        '.md':TextLoader
    }
    documents = []
    document_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in supported_extensions:
                file_path = os.path.join(root, file)
                try:
                    loader_class = supported_extensions[file_ext]
                    if file_ext in ['.txt', '.md']:
                        for encoding in ['utf-8', 'gbk', 'gb2312']:
                            try:
                                loader = loader_class(file_path, encoding = encoding)
                                docs = loader.load()
                                documents.extend(docs)
                                document_paths.append(file_path)
                                print(f"成功加载文件：{file_path} 编码：{encoding}")
                                break
                            except Exception as e:
                                print(f"无法解码文件：{file_path} : {e}")
                    else:
                        loader = loader_class(file_path)
                        docs = loader.load()
                        documents.extend(docs)
                        document_paths.append(file_path)
                        print(f"成功加载: {file_path}")
                except Exception as e:
                    print(f"加载文件失败：{file_path} ：{e}")
    
    return documents, document_paths
def load_documents_from_path(input_path):
    """根据输入路径加载文档（支持文件和文件夹）"""
    if os.path.isfile(input_path):
        # 单个文件
        file_ext = os.path.splitext(input_path)[1].lower()
        if file_ext == '.pdf':
            loader = PyPDFLoader(input_path)
        else:
            for encoding in ['utf-8', 'gbk', 'gb2312']:
                try:
                    loader = TextLoader(input_path)
                    break
                except Exception as e:
                    continue
            else:
                raise ValueError(f"无法解码文件：{input_path}")
            
        documents = loader.load()
        document_paths = [input_path]
    elif os.path.isdir(input_path):
        documents, document_paths = load_documents_from_folder(input_path)
    else:
        raise ValueError(f"路径不存在： {input_path}")
    
    print(f"共加载{len(documents)}个文档片段，来自{len(document_paths)}个文件。")
    return documents, document_paths

def debug_write_chunks_to_file(texts, filename="text_chunks_debug.txt"):
    """将文本块写入文件，避免控制台显示问题"""
    with open(filename, 'w', encoding='utf-8') as f:
        for i, text in enumerate(texts):
            f.write(f"=== 文本块 {i+1} ===\n")
            f.write(text.page_content)
            f.write("\n" + "="*50 + "\n\n")
    print(f"文本块已写入文件: {filename}")
def setup_vector_store(input_path, embedding_model_path, vectorstore_path = "./local_chroma_db"):
    """设置向量存储
    参数 input_path 文档路径或文件夹路径
         embedding_model_path 嵌入模型的路径
         vectorestore_path 向量存储路径
    返回 vectorstore:向量存储对象
         document_hash 文档hash值
    """
    print("检查文档状态")
    document_hash = get_input_path_hash(input_path)
    recreate_needed, reason = should_recreate_vectorstore(vectorstore_path, document_hash)
    if recreate_needed:
        print(f"需要重新创建向量存储数据：{reason}")
        documents, document_paths = load_documents_from_path(input_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap = 50,
            length_function=len,
            separators=["\n# ", "\n## ", "\n### ", "\n#### ", "\n- ", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        print(f"====文档已分割为{len(texts)}个文本块====")
        debug_write_chunks_to_file(texts, "text_chunks_debug.txt")
        embeddings = HuggingFaceEmbeddings(
            model_name = embedding_model_path,
            model_kwargs = {'device':'cuda'},
            encode_kwargs = {'normalize_embeddings':False}
        )
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=vectorstore_path,
            collection_metadata={"hnsw:space": "cosine"}  # 显式指定余弦相似度 这里可以将分数归一到0-1
        )
        # 保存元数据
        save_vectorstore_metadata(vectorstore_path, document_hash, document_paths)
        print("====向量存储创建完成并已缓存====")
    else:
        print("====加载已经缓存的vectorstore====")
        embeddings = HuggingFaceEmbeddings(
            model_name = embedding_model_path,
            model_kwargs={'device':'cuda'},
            encode_kwargs={'normalize_embeddings':False}
        )
        vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}  # 显式指定余弦相似度
        )
        print("====向量存储加载完成====")
    return vectorstore, document_hash

# 在 vector_store_manager.py 中添加向量存储验证
def validate_vector_store(vectorstore, test_questions):
    """验证向量存储的质量"""
    print("\n====验证向量存储质量====")
    
    for question in test_questions:
        print(f"\n测试问题: '{question}'")
        
        # 相似度搜索
        docs = vectorstore.similarity_search(question, k=2)
        for i, doc in enumerate(docs):
            relevance = "✓" if any(keyword in doc.page_content for keyword in question) else "✗"
            print(f"  结果 {i+1} {relevance}: {doc.page_content[:80]}...")
    
    # 检查向量存储中的文档数量
    collection = vectorstore._collection
    if hasattr(collection, 'count'):
        count = collection.count()
        print(f"\n向量存储中的文档数量: {count}")