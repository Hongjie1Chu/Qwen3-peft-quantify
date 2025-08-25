# 🌟 Qwen3 模型微调、量化与部署工具链 🚀

## 项目简介

**Qwen3 微调、量化与部署工具链** 是一个专为 Qwen3 模型设计的高效微调与量化解决方案。本项目通过 LoRA 参数高效微调技术和 AWQ 4-bit 量化技术，帮助开发者在有限资源条件下，快速定制和部署高性能大语言模型。无论是个人研究者还是企业开发者，都能通过本工具链显著降低模型训练和部署门槛，实现从数据准备、模型微调到量化部署的完整工作流。项目特别优化了内存使用和训练效率，使大规模的模型能够在消费级 GPU 上高效运行。
#### **让大模型定制像搭积木一样简单！**

你是否也觉得训练大模型门槛太高？内存爆了、显存不够、部署复杂……别担心！  
我们为你打造了一套 **轻量、高效、一键式** 的 Qwen3 微调 + 量化 + 部署全流程工具链！

基于 **LoRA 参数高效微调** 和 **AWQ 4-bit 量化技术**，即使在消费级 GPU 上，也能轻松玩转 Qwen3-0.6B！  
从数据准备到服务上线，一气呵成，专为开发者、研究者和“想试试大模型”的你而生 💡

---

## 📁 项目结构一览

```bash
.
├── datasets/                  # 你的数据小窝
│   └── peft_data.jsonl
├── download_model.py          # 一键下载模型
├── inference/                 # 推理 & 部署全家桶
│   ├── deploy_vllm.sh         # 快速部署脚本
│   ├── infer_offline.ipynb    # 离线测试笔记本
│   └── send_req.py            # 发送请求小助手
├── install.sh                 # 环境搭建“魔法咒语”
├── peft/
│   └── LoRA/
│       └── lora.py            # LoRA微调核心脚本
├── quant/
│   ├── AWQ/
│   │   └── auto_awq_quantify.py  # 4-bit量化，省显存利器
│   └── GPTQ_MODEL/
│       └── gptq_model_quantify.py
└── readme.md                  # 就是现在看的这个啦～
```

---

## ⚙️ 快速上手：三步走，起飞！🚀

### 1. 环境准备 💻

```bash
# 创建专属环境
conda create -n vllm python=3.10
conda activate vllm

# 安装依赖（一键搞定）
sh install.sh
```

> 💡 提示：PyTorch 版本请根据你的 GPU 选择，[官网安装指南](https://pytorch.org/get-started/locally/) 帮你精准匹配！

---

### 2. 模型定制三连击 🔧

#### 📥 下载模型
```bash
python download_model.py
```

#### 🔧 微调模型（LoRA）
```bash
python peft/LoRA/lora.py
```
✅ 输出：
- **轻量适配器**：`output/peft_model/Qwen3-0.6B-lora`
- **完整模型**：`output/full_model/Qwen3-0.6B`（可直接部署）

#### ⚡ 量化压缩（AWQ 4-bit）
```bash
python quant/AWQ/auto_awq_quantify.py
```
✅ 输出：
- **4-bit 量化模型**：`output/quantized_model/Qwen3-0.6B-awq-4bit`  
  显存占用 ↓↓↓，推理速度 ↑↑↑，部署更轻松！

---

### 3. 部署上线，开始对话！🌐

#### 🚀 一键部署（推荐）
```bash
sh inference/deploy_vllm.sh
```

或手动启动：
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/Qwen3-peft-quantify/output/quantized_model/Qwen3-0.6B-awq-4bit \
    --port 7890 \
    --enable-reasoning \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
```

#### 📬 发送请求试试看！

- **关闭思考模式（简洁回复）**
  ```bash
  python inference/send_req.py "你好" False
  ```

```
(lora_quant) root@autodl-container-f9074882c7-d31e3610:~/autodl-tmp/Qwen3-peft-quantify/inference# python send_req.py  "你好" False
用户提问: 你好
思考模式: False
模型回复:
你好！有什么可以帮助你的吗？
```

- **开启思考模式（深度推理）**
```bash
python inference/send_req.py "你好" True
```
  
  ```
(lora_quant) root@autodl-container-f9074882c7-d31e3610:~/autodl-tmp/Qwen3-peft-quantify/inference# python send_req.py  "你好" False
用户提问: 你好
思考模式: False
模型回复:
你好！有什么可以帮助你的吗？
  ```

---

## ❓ 常见问题 & 解决方案

🔧 **问题1：校准数据不足报错**
```
RuntimeError: torch.cat(): expected a non-empty list of Tensors
```
✅ **解决方法**：减小 `max_calib_seq_len` 参数（如设为 `32` 或 `64`），小数据集更友好！

🔧 **问题2：Transformers 兼容性报错**
```
AttributeError: 'Catcher' object has no attribute 'attention_type'
```
✅ **解决方法**：降级 Transformers 库
```bash
pip uninstall transformers -y
pip install transformers==4.51.3
```

---

✨ **一句话总结**：  
**下载 → 微调 → 量化 → 部署**，四步闭环，轻松拥有你的专属 Qwen3 模型！  
无论是实验、测试还是轻量级应用，这套工具链都能帮你省时省力，快速落地 🎉

> 💌 项目持续更新中，欢迎 Star ⭐ & 提 Issue 交流！一起打造更酷的大模型工具链！

--- 
