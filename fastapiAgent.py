from fastapi import FastAPI, Depends
from pydantic import BaseModel
from agent0422 import get_answer
import uvicorn
from model.res import Response
app = FastAPI()

class Query(BaseModel):
    question: str
# uvicorn fastapiAgent:app --reload
@app.get("/wkh")
async def wkh(query: Query = Depends()):
    answer = get_answer(query.question)
    return {"message": answer}

@app.post("/tomato")
async def tomato(query: Query):
    answer = get_answer(query.question)
    return Response.ok(answer)

from fastapi.responses import StreamingResponse
from agent0422 import stream_answer
class ChatRequest(BaseModel):
    question: str
    thread_id: str = "1"  # 支持多会话

@app.post("/agent/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口"""
    return StreamingResponse(
        stream_answer(request.question, request.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/agent/generate")
async def generate():
    # 流式响应示例
    return {"message": "Hello, World!"}

if __name__ == "__main__":
    # 直接在代码中启动，而不是用命令行 uvicorn
    uvicorn.run(
        "fastapiAgent:app",  # ← 改为字符串形式
        host="127.0.0.1",
        port=8000,
        log_level="info"  # 可选：控制日志详细程度
    )