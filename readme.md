# Qwen3 模型微调、量化与部署 🚀

## 项目简介

**Qwen3 微调、量化与部署工具链** 是一个专为 Qwen3 模型设计的高效微调与量化解决方案。本项目通过 LoRA 参数高效微调技术和 AWQ 4-bit 量化技术，帮助开发者在有限资源条件下，快速定制和部署高性能大语言模型。无论是个人研究者还是企业开发者，都能通过本工具链显著降低模型训练和部署门槛，实现从数据准备、模型微调到量化部署的完整工作流。项目特别优化了内存使用和训练效率，使大规模的模型能够在消费级 GPU 上高效运行。

## 📁 项目结构

```
├── requirements.txt # 依赖文件 📦
├── peft/lora.py # 微调脚本目录 🔧
├── quant/AWQ/auto_awq_quantify # 量化脚本目录 ⚡
└── datasets/peft_data.jsonl # 训练数据集 📊
```

## ⚙️ 快速开始

### 1. 环境搭建 💻
- 搭建环境
```bash
conda create -n vllm python=3.10
conda activate vllm
pip install -r requirements.txt
```

- 下载模型-微调-量化
```
python download_model
python lora.py 
python auto_awq_quantify.py
```

- 服务化部署
```bash
python -m vllm.entrypoints.openai.api_server \
--model /root/autodl-tmp/Qwen3-peft-quantify/output/quantized_model/Qwen3-0.6B-awq-4bit \
--port 8000 \
--gpu-memory-utilization 0.8 \  
--max-num-seqs 8 \
```

发送请求
```
python send_request.py
```
### 2. 微调模型 🔧

- **脚本位置**: `peft/LoRA/lora.py`

- **功能**: 对Qwen3-0.6B进行高效LoRA微调并自动保存合并模型 🔄

- **输出**:

- LoRA适配器: `output/peft_model/Qwen3-0.6B-lora` (仅微调参数)

- 完整模型: `output/full_model/Qwen3-0.6B` (可直接部署)

### 3. 量化模型 ⚡

- **脚本位置**: `quant/AWQ/auto_awq_quantify.py`

- **功能**: 对微调合并后的模型进行4-bit AWQ量化，大幅降低资源需求 🔋

- **输出**: `output/quantized_model/Qwen3-0.6B-awq-4bit` (4-bit量化模型)

### 4. 常见问题与解决方案 ❓

1. **校准数据不足报错**:
```
RuntimeError: torch.cat(): expected a non-empty list of Tensors
```
✅ **解决方案**: 修改量化参数 `max_calib_seq_len`，数据集较小时请减小这个值 (建议设置为64或32)

2. **Transformer版本兼容性问题**:
```
AttributeError: 'Catcher' object has no attribute 'attention_type'
```
✅ **解决方案**: 降级Transformers库版本
```bash
pip uninstall transformers -y
pip install transformers==4.51.3
```

> 💡 **提示**: PyTorch安装请根据您的GPU环境选择合适版本，参考[PyTorch官网](https://pytorch.org/get-started/locally/)获取安装命令

---

✨ 简单高效的Qwen3模型微调与量化方案，助您轻松部署大语言模型！