import os
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.tools import tool
#短期记忆
from langgraph.checkpoint.memory import InMemorySaver
from tools.langsmith import setup_langsmith_env

setup_langsmith_env()

@tool("get_author",description="当用户问你的作者是谁时，调用该工具")
def get_author(question: str) -> str:
    """根据问题获取作者。"""
    return f"当然是tomato!"
def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"
model = ChatOpenAI(
    model="qwen-plus", # 模型名：qwen-turbo, qwen-plus, qwen-max
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # 必须指定阿里的兼容接口地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

agent = create_agent(
    model=model,
    tools=[get_weather,get_author],
    checkpointer=InMemorySaver(),  # [!code highlight]
    system_prompt="你是一个招投标平台的智能客服，使用中文回答问题。你的用户多数为评标专家和招标代理。回答问题要谨慎，不知道的问题不会要胡编乱造，可以提示转人工技术回答。问你可以做什么时，回答自己是招投标智能助手。使用客服的语气回答问题。",
)

# 运行代理
# response = agent.invoke(
#     human_msg
# )
# print(response)

app = FastAPI(title="Qwen-Plus RAG 智能客服（稳定版）")

class ChatRequest(BaseModel):
    question: str

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        print(req.question)
        result = agent.invoke(
            {"messages": [HumanMessage(req.question)]},
            {"configurable": {"thread_id": "1"}}
        )
        return {
            "code": 200,
            "question": req.question,
            "answer": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "服务启动成功！"}