python -m vllm.entrypoints.openai.api_server \
--model /root/autodl-tmp/Qwen3-peft-quantify/output/quantized_model/Qwen3-0.6B-awq-4bit \
--port 7890 \
--gpu-memory-utilization 0.8 \
--max-num-seqs 8