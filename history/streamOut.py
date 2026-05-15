#提示词
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
import os
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages


#提示词
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个智能客服，请回答用户问题。"),
    #占位符，把状态图里的messages变量放到这个位置，变量名必须是messages
    MessagesPlaceholder(variable_name="messages"),
])


#定义一个状态图
workflow = StateGraph(MessagesState)

model = ChatOpenAI(
    model="qwen-plus", # 模型名：qwen-turbo, qwen-plus, qwen-max
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # 必须指定阿里的兼容接口地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
import tiktoken
def count_tokens_in_messages(messages):

    """估算 messages 的总 token 数（参考 OpenAI 官方方式）"""
    encoding = tiktoken.get_encoding("cl100k_base")  # Qwen 使用此编码
    tokens_per_message = 4  # 每条消息的固定开销（role + 分隔符等）
    tokens_per_name = -1    # Qwen 不支持 name 字段，可忽略

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        num_tokens += len(encoding.encode(message.content))
        num_tokens += len(encoding.encode(message.type))  # 'system', 'user', 'assistant'
        # 注意：Qwen 不支持 'name'，所以跳过
    num_tokens += 3  # 每轮对话的额外开销（如 <|im_start|> 等）
    return num_tokens

trimmer = trim_messages(
    max_tokens=650, #最大 token 数量限制
    strategy="last",#裁剪策略：保留最近的消息 也就是：删最早的，留最新的
    token_counter=count_tokens_in_messages,#让模型自己算自己的 token，最准确
    include_system=True, #系统消息算进 token 数量，并且一定保留  false系统提示可能被裁掉
    allow_partial=False, #不允许截断单条消息 要么整条保留，要么整条删掉
    start_on="human", #裁剪后的第一条消息必须是用户消息
)


# 定义一个函数，用于调用模型并返回响应
def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state["messages"])
    # 把提示词和用户消息拼在一起
    prompt = prompt_template.invoke({"messages": trimmed_messages})
    # 执行加了提示词后的用户消息
    response = model.invoke(prompt)
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

query = "我是tomato."


input_messages = [HumanMessage(query)]
output = app.stream({"messages": input_messages}, config,stream_mode="messages",)
for chunk, metadata in output:
    print(chunk.content, end="-")



