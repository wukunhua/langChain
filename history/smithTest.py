# from tools.langsmith import setup_langsmith_env

# setup_langsmith_env()
import os
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
messages = [
    SystemMessage("Translate the following from English into Chinese"),
    HumanMessage("hi!,what is your name?"),
]


model = ChatOpenAI(
    model="qwen-plus", # 模型名：qwen-turbo, qwen-plus, qwen-max
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # 必须指定阿里的兼容接口地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

for chunk in model.stream(messages):
    print(chunk.content, end="-")
