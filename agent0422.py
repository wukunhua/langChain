#究极版本

#构建llm大模型
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
model = ChatOpenAI(
    model="qwen-plus", # 模型名：qwen-turbo, qwen-plus, qwen-max
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # 必须指定阿里的兼容接口地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

#创建智能体
from langchain.agents import create_agent
#引入ragtools
from tools.ragTools import retrieve_context
from tools.weatherTools import get_weather
#引入system_instruction
system_instruction = """你是一个专业的智能助手，负责回答关于公司内部知识库的问题。
你的核心规则是：
1.  当用户询问任何业务、产品、流程相关问题时，**必须且只能**使用 `retrieve_context` 工具来获取答案，不要根据自己的知识回答。
2.  如果 `retrieve_context` 工具返回的结果为空或表示不知道，请礼貌地告诉用户：“知识库中暂无相关信息，请联系人工支持。”
3.  请使用中文回答。
4.  回答时请保持简洁、专业。
5.  如果用户提出关于天气的问题，请使用 `get_weather` 工具来获取答案。
"""
agent = create_agent(
    model,
    [retrieve_context,get_weather],
    # agent_type="zero-shot-react-description",
    # verbose=True,
    system_prompt=system_instruction
)
#question = "机器学习可以分为哪三类？"
# question = "你是谁？你能做什么？今天天气怎么样？"
# response = agent.invoke(
#     {"messages": [HumanMessage(question)]},
#     {"configurable": {"thread_id": "1"}}
# )

def get_answer(question):
    return agent.invoke(
        {"messages": [HumanMessage(question)]},
        {"configurable": {"thread_id": "1"}}
    )["messages"][-1].content
import json
async def stream_answer(question: str, thread_id: str):
    """生成流式回答"""
    try:
        async for event in agent.astream_events(
            {"messages": [HumanMessage(question)]},
            {"configurable": {"thread_id": thread_id}},
            version="v2"
        ):
            # 处理不同类型的流式事件
            if event["event"] == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    # 以 SSE 格式输出
                    yield f"data: {json.dumps({'token': content}, ensure_ascii=False)}\n\n"
            
            elif event["event"] == "on_chain_end":
                if event["name"] == "Agent":
                    # 输出结束标记
                    yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

#print(response)
#print(response["messages"][-1].content)


#RAG部分

