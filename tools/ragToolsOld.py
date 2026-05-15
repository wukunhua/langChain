from langchain_community.document_loaders import PyPDFLoader, TextLoader
#加载pdf文件
file_path = "./knowledge/RAG.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

#分割文档
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 每一小段最长 1000 个字符
    chunk_overlap=20,  # 段与段之间重叠 200 个字符 上一段的最后 200 字，会重复出现在下一段开头
    add_start_index=True,  # 记录每一段在原文的开始位置
)
all_splits = text_splitter.split_documents(docs)
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
#  嵌入模型（DashScope，纯阿里）
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 最新、最强
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY") # 可显式传入
)

# 向量库
vectorstore = FAISS.from_documents(all_splits, embeddings) #把文本变成向量，存进内存向量库。
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) #从向量库中检索2条文档交给大模型处理 给的多了大模型容易混乱
#RAG工具
from langchain.tools import tool
#content给大模型，artifact给智能体
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = retriever.invoke(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs