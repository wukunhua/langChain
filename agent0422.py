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
#引入system_instruction
system_instruction = """你是一个专业的智能助手，负责回答关于公司内部知识库的问题。
你的核心规则是：
1.  当用户询问任何业务、产品、流程相关问题时，**必须且只能**使用 `retrieve_context` 工具来获取答案，不要根据自己的知识回答。
2.  如果 `retrieve_context` 工具返回的结果为空或表示不知道，请礼貌地告诉用户：“知识库中暂无相关信息，请联系人工支持。”
3.  回答时请保持简洁、专业。
"""
agent = create_agent(
    model,
    [retrieve_context],
    # agent_type="zero-shot-react-description",
    # verbose=True,
    system_prompt=system_instruction
)
question = "机器学习可以分为哪三类？"
response = agent.invoke(
    {"messages": [HumanMessage(question)]},
    {"configurable": {"thread_id": "1"}}
)
print(response)
print(response["messages"][-1].content)


#RAG部分

