from openai import OpenAI
import requests

# 修改 OpenAI 的 API 密钥和 API 基础 URL 以使用 vLLM 的 API 服务器。
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:7890/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
def get_response(prompt):
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="/root/autodl-tmp/Qwen3-peft-quantify/output/quantized_model/Qwen3-0.6B-awq-4bit",
        messages=[
            {"role": "user", "content": prompt},
        ],
        # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
        # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}  # ✅ 严格关闭思考
        }
    )
    return completion.choices[0].message.content
print(get_response('你是谁?'))
