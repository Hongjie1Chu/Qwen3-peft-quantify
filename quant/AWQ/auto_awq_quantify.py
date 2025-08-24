#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Qwen3-0.6B 模型量化脚本 (AWQ)
功能特点：
1. 支持完全自定义路径配置
2. 智能校准数据集处理
3. 清晰的量化流程管理
"""

import os
import json
import random
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def load_calibration_data(calib_file, tokenizer, num_samples=10, seed=42):
    """加载并处理校准数据集"""
    print(f"🔄 加载校准数据集: {calib_file}...")
    
    # 检查文件是否存在
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"校准数据文件不存在: {calib_file}")
    
    # 读取JSONL文件
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    
    # 解析JSONL
    data_list = [json.loads(line.strip()) for line in lines]
    
    # 随机选择指定数量的样本
    print(f"  • 总样本数: {len(data_list)}，将随机选择 {num_samples} 条用于校准")
    random.seed(seed)
    selected_data = random.sample(data_list, min(num_samples, len(data_list)))
    
    # 转换为Qwen对话格式
    dataset = []
    for item in selected_data:
        messages = [
            {"role": "user", "content": item['input']},
            {"role": "assistant", "content": item['output']}
        ]
        dataset.append(messages)
    
    # 应用tokenizer的chat template
    print("  • 应用tokenizer的chat template...")
    data = []
    for msg in dataset:
        text = tokenizer.apply_chat_template(
            msg, 
            tokenize=False, 
            add_generation_prompt=False
        )
        data.append(text.strip())
    
    print(f"✅ 校准数据处理完成，共 {len(data)} 条有效数据")
    return data


def quantize_model(model_path, quant_path, calib_data, quant_config):
    """执行模型量化"""
    print(f"⚡ 开始模型量化 (路径: {model_path} → {quant_path})...")
    
    # 加载模型
    print(f"  • 加载模型: {model_path}")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        safetensors=True
    )
    
    # 获取tokenizer
    print(f"  • 加载tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 执行量化
    print("  • 开始量化过程...")
    print(f"  • 量化配置: {quant_config}")
    
    model.quantize(
        tokenizer, 
        quant_config=quant_config,
        calib_data=calib_data,
        max_calib_seq_len=128  # 校准数据太少时，设置小一点
    )
    
    # 保存量化后的模型
    print(f"  • 保存量化模型至: {quant_path}")
    os.makedirs(quant_path, exist_ok=True)
    model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
    tokenizer.save_pretrained(quant_path)
    
    print(f"✅ 模型量化完成并保存至: {quant_path}")
    return quant_path


def main():
    """主量化流程 - 所有路径配置在此处自定义"""
    print("✨ 开始配置量化参数...")
    
    # ======================
    # 用户自定义路径配置区
    # ======================
    
    # 基础模型
    BASE_MODEL_NAME = 'Qwen3-0.6B'
    # 输出目录配置
    OUTPUT_BASE_DIR = "../../output"
    
    # 模型配置
    MERGED_MODEL_PATH = f"{OUTPUT_BASE_DIR}/merged_model/{BASE_MODEL_NAME}/lora_merged_model"  # 待量化的合并模型路径
    
    # 数据集配置
    CALIBRATION_DATASET = "../../datasets/peft_data.jsonl"  # 校准数据集路径
    
    # 量化路径
    QUANTIZED_MODEL_DIR = f"{OUTPUT_BASE_DIR}/quantized_model/{BASE_MODEL_NAME}-awq-4bit"
    
    # ======================
    # 量化参数配置区
    # ======================
    
    QUANT_CONFIG = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }
    
    CALIBRATION_SAMPLES = 10  # 用于校准的样本数量
    RANDOM_SEED = 42          # 随机种子，确保可重现
    
    # ======================
    # 路径配置结束
    # ======================
    
    print("\n📌 自定义配置:")
    print(f"  • 待量化模型路径: {MERGED_MODEL_PATH}")
    print(f"  • 校准数据集路径: {CALIBRATION_DATASET}")
    print(f"  • 量化后模型保存路径: {QUANTIZED_MODEL_DIR}")
    print(f"  • 量化配置: {QUANT_CONFIG}")
    print(f"  • 校准样本数量: {CALIBRATION_SAMPLES}")
    
    # 步骤1: 检查模型路径是否存在
    print("\n🔍 验证模型路径...")
    if not os.path.exists(MERGED_MODEL_PATH):
        raise FileNotFoundError(f"模型路径不存在: {MERGED_MODEL_PATH}")
    print(f"  • 检测到模型路径: {MERGED_MODEL_PATH}")
    
    # 步骤2: 加载tokenizer用于校准数据处理
    print("\n🔄 加载tokenizer以处理校准数据...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"无法加载tokenizer: {str(e)}")
    
    # 步骤3: 加载并处理校准数据
    print("\n📊 准备校准数据...")
    calib_data = load_calibration_data(
        CALIBRATION_DATASET, 
        tokenizer,
        num_samples=CALIBRATION_SAMPLES,
        seed=RANDOM_SEED
    )
    
    # 步骤4: 执行模型量化
    print("\n⚡ 执行模型量化...")
    quantize_model(
        model_path=MERGED_MODEL_PATH,
        quant_path=QUANTIZED_MODEL_DIR,
        calib_data=calib_data,
        quant_config=QUANT_CONFIG
    )
    
    print("\n🎉 量化流程全部完成!")


if __name__ == "__main__":
    # 设置CUDA可见设备
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    main()