from modelscope import snapshot_download
# 在modelscope上下载Qwen模型到本地目录下
model_path = snapshot_download("Qwen/Qwen3-0.6B", cache_dir="/root/autodl-tmp/", revision="master")