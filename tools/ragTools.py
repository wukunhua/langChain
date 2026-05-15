from langchain_community.document_loaders import PyPDFLoader, TextLoader,UnstructuredWordDocumentLoader
from pathlib import Path
import chardet
#加载pdf文件
KNOWLEDGE_DIR = "./knowledge"
def load_text_file(file_path):
    """自动检测编码并加载文件"""
    # 检测文件编码
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
        print(f"检测到编码: {encoding}")
    
    # 使用检测到的编码加载
    loader = TextLoader(file_path, encoding=encoding)
    return loader
def load_all_documents(directory_path: str):
    """加载文件夹中的所有文档"""
    all_docs = []
    
    # 支持的文档格式及其对应的加载器
    loaders_map = {
        ".pdf": PyPDFLoader,
        ".txt": load_text_file,
        ".docx": UnstructuredWordDocumentLoader,
        # 可以继续添加其他格式
        # ".md": TextLoader,
    }
    
    # 遍历文件夹中的所有文件
    for file_path in Path(directory_path).iterdir():
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            if suffix in loaders_map:
                try:
                    print(f"正在加载: {file_path.name}")
                    loader_class = loaders_map[suffix]
                    loader = loader_class(str(file_path))
                    docs = loader.load()
                    all_docs.extend(docs)
                    print(f"  ✅ 成功加载 {len(docs)} 个文档片段")
                except Exception as e:
                    print(f"  ❌ 加载失败 {file_path.name}: {e}")
            else:
                print(f"  ⏭️  跳过不支持的文件格式: {file_path.name}")
    
    print(f"\n总计加载 {len(all_docs)} 个文档片段")
    return all_docs


# file_path = "./knowledge/RAG.pdf"
# loader = PyPDFLoader(file_path)

docs = load_all_documents(KNOWLEDGE_DIR)

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