import bs4
import os
#from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langgraph.graph import START, StateGraph
# from typing_extensions import List, TypedDict

# Load and chunk contents of the blog
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
#找页面上class等于post-title,post-header,post-content的元素
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()


# 对获取到的文档进行分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 每一小段最长 1000 个字符
    chunk_overlap=200,  # 段与段之间重叠 200 个字符 上一段的最后 200 字，会重复出现在下一段开头
    add_start_index=True,  # 记录每一段在原文的开始位置
)
all_splits = text_splitter.split_documents(docs)
print(f"Total characters: {len(docs[0].page_content)}")#一共多少字符
print(f"Split blog post into {len(all_splits)} sub-documents.")#分成了多少小段






