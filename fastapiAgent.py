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


@app.post("/generate")
async def generate(query: Query):
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