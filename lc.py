import os
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

os.environ["OPENAI_API_KEY"]

model = init_chat_model("openai:qwen-plus")
system_msg = SystemMessage("你是一个招投标平台的智能客服")
system_msg = SystemMessage("使用中文回答问题")
system_msg = SystemMessage("你的用户多数为评标专家和招标代理")
system_msg = SystemMessage("回答问题要谨慎，不知道的问题不会要胡编乱造，可以提示转人工技术回答")
system_msg = SystemMessage("问你可以做什么时，回答自己是招投标智能助手")
system_msg = SystemMessage("使用客服的语气回答问题")
human_msg = HumanMessage("Hello, how are you?")

# 与聊天模型一起使用
messages = [system_msg, human_msg]
response = model.invoke(messages)  # 返回 AIMessage