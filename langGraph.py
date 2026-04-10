#内存持久化
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
import os
from langchain.messages import HumanMessage, AIMessage, SystemMessage
#定义一个状态图
workflow = StateGraph(MessagesState)

model = ChatOpenAI(
    model="qwen-plus", # 模型名：qwen-turbo, qwen-plus, qwen-max
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # 必须指定阿里的兼容接口地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 定义一个函数，用于调用模型并返回响应
def call_model(state: MessagesState):
    print(state)

    response = model.invoke(state['messages'])
    return {"messages": response}

# 添加边 开始节点到模型节点
workflow.add_edge(START, "model")
# 添加节点 模型节点调用callmodel函数
workflow.add_node("model", call_model)
# 使用内存持久化状态图
memory = MemorySaver()
# 编译状态图
app = workflow.compile(checkpointer=memory)
# 配置内存持久化id
config = {"configurable": {"thread_id": "abc123"}}

query = "Hi! I'm Bob."


input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()  # output contains all messages in state

query1 = "我是谁"
input_messages1 = [HumanMessage(query1)]
output = app.invoke({"messages": input_messages1}, config)
output["messages"][-1].pretty_print()  # output contains all messages in state