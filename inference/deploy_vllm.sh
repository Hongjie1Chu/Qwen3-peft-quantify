# 启动 vLLM 时务必加 --enable-reasoning
python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/Qwen3-peft-quantify/output/quantized_model/Qwen3-0.6B-awq-4bit \
    --port 7890 \
    --enable-reasoning \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096