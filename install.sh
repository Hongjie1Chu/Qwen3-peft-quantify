pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install vllm==0.9.2 # 用于服务化部署
pip install modelscope==1.25.0 # 用于模型下载和管理
pip install transformers==4.51.3 # Hugging Face 的模型库，用于加载和训练模型
pip install accelerate==1.6.0 # 用于分布式训练和混合精度训练
pip install datasets==3.5.1 # 用于加载和处理数据集
pip install peft==0.15.2 # 用于 LoRA 微调
pip install swanlab==0.5.7 # 用于监控训练过程与评估模型效果
pip install autoawq-kernels autoawq # 用于awq量化