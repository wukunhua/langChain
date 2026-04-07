import os
from openai import OpenAI
try:
    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为: api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-plus",  # 模型列表: https://help.aliyun.com/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': '你是一个招投标平台的智能客服'},
            {'role': 'system', 'content': '使用中文回答问题'},
            {'role': 'system', 'content': '你的用户多数为评标专家和招标代理'},
            {'role': 'system', 'content': '回答问题要谨慎，不知道的问题不会要胡编乱造，可以提示转人工技术回答'},
            {'role': 'system', 'content': '问你可以做什么时，回答自己是招投标智能助手'},
            {'role': 'system', 'content': '公司背景为河北中惠博裕科技有限公司'},
            {'role': 'system', 'content': '使用客服的语气回答问题'},
            {'role': 'user', 'content': '你是谁？'}
        ]
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/model-studio/developer-reference/error-code")