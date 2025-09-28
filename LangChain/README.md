# main_huggingface.py是一个主运行程序，里面包含了使用LangChain的例子

# main_qwen.py 此文件展示了使用qwen的模型来展示LangChain中的chain功能。实现后，将代码转移到了main_huggingface.py中了

# main_openai.py 此文件则是用openai接口的方式

***下边的是一伙的 ***
# main_RAG 此文件主要是展示RAG的例子。读取外部的pdf等格式的文件后，分片文本，嵌入，向量化，问答
# 2025-09-25 新增文件
## ModelManager_RAG.py是负责模型加载的模块
## VectorStoreManager_RAG.py是负责文档加载和向量存储的模块