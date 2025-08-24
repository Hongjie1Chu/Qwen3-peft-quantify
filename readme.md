# Qwen3模型微调与量化 🚀

## 📁 项目结构

```
.
├── requirements.txt    # 依赖文件 📦
├── peft/               # 微调脚本目录 🔧
├── quant/              # 量化脚本目录 ⚡
└── datasets/peft_data.jsonl # 训练数据集 📊
```

## ⚙️ 快速开始

### 1. 环境搭建 💻
```bash
conda create -n qwen3 python=3.10
conda activate qwen3
pip install -r requirements.txt
```

### 2. 微调模型 🔧
- **脚本位置**: `peft/LoRA/lora.py`
- **功能**: 对Qwen3-0.6B进行高效LoRA微调并自动保存合并模型 🔄
- **输出**: 
  - LoRA适配器: `output/peft_model/Qwen3-0.6B-lora` (仅微调参数)
  - 完整模型: `output/full_model/Qwen3-0-6B` (可直接部署)

### 3. 量化模型 ⚡
- **脚本位置**: `quant/AWQ/quantify.py`
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