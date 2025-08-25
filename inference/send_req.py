from openai import OpenAI
import sys

# 配置本地 vLLM 服务
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:7890/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def get_response(prompt="你是谁？", enable_thinking=False):
    """
    调用本地 vLLM 部署的 Qwen3 模型获取回复

    参数:
        prompt (str): 输入提示，默认为“你是谁？”
        enable_thinking (bool): 是否开启思考模式，默认关闭

    返回:
        str: 模型生成的回复
    """
    try:
        completion = client.chat.completions.create(
            model="/root/autodl-tmp/Qwen3-peft-quantify/output/quantized_model/Qwen3-0.6B-awq-4bit",
            messages=[
                {"role": "user", "content": prompt},
            ],
            extra_body={
                "chat_template_kwargs": {"enable_thinking": enable_thinking, 'min_tokens':128}
            },
            # min_tokens=32  # ✅ 设置最小生成长度为 32 tokens
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"请求失败: {e}"

if __name__ == "__main__":
    # 解析命令行参数：python sent_request.py "prompt" "True/False"
    prompt = sys.argv[1] if len(sys.argv) > 1 else "你是谁？"
    enable_thinking_str = sys.argv[2] if len(sys.argv) > 2 else "False"

    # 转换字符串为布尔值
    enable_thinking = enable_thinking_str.strip().lower() in ("true", "1", "yes", "on")

    # 调用函数并打印结果
    response = get_response(prompt, enable_thinking)
    print("用户提问:", prompt)
    print("思考模式:", enable_thinking)
    print("模型回复:\n", response)