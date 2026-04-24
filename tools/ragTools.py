from langchain_community.document_loaders import PyPDFLoader
#加载pdf文件
file_path = "./knowledge/RAG.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

# print(len(docs[0].page_content))


# 人工智能的核心领域包括哪些？
# 机器学习可以分为哪三类？
# 基于 Transformer 架构的预训练语言模型，提升了自然语言处理的哪些任务效果？

#分割文档
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 每一小段最长 1000 个字符
    chunk_overlap=20,  # 段与段之间重叠 200 个字符 上一段的最后 200 字，会重复出现在下一段开头
    add_start_index=True,  # 记录每一段在原文的开始位置
)
all_splits = text_splitter.split_documents(docs)

#print(len(all_splits))

# 嵌入文档
import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
# 1. 嵌入模型（DashScope，纯阿里）
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 最新、最强
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY") # 可显式传入
)
# 2. LLM：qwen-plus（用 OpenAI 兼容接口）
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 同一个 Key！
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.1
)
# 5. 向量库
vectorstore = FAISS.from_documents(all_splits, embeddings) #把文本变成向量，存进内存向量库。
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) #从向量库中检索2条文档交给大模型处理 给的多了大模型容易混乱

   
# 构建 RAG 链
# ----------------------
# ✅ 新版 RAG 链（无报错，替代旧 chains）
# ----------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 提示词模板
template = """根据下面的上下文回答问题：
{context}

问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 构建 RAG 链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 提问
if __name__ == "__main__":
    question = "机器学习可以分为哪三类？"
    result = rag_chain.invoke(question)
    print("回答：", result)