#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Qwen3-0.6B GPTQ量化脚本
功能特点：
1. 与项目路径配置完全兼容
2. 简化校准数据处理流程
3. 清晰的量化过程反馈
"""

import os
import json
import torch
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer


def load_calibration_data(calib_file, tokenizer, n_samples=100, seq_len=1024):
    """加载并处理校准数据集"""
    print(f"🔄 加载校准数据集: {calib_file}...")
    
    # 检查文件是否存在
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"校准数据文件不存在: {calib_file}")
    
    # 读取JSONL文件
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    
    # 随机选择样本
    import random
    random.seed(42)
    selected_lines = random.sample(lines, min(n_samples, len(lines)))
    
    # 处理校准数据
    calibration_dataset = []
    skipped = 0
    
    for line in selected_lines:
        try:
            item = json.loads(line.strip())
            
            # 构建Qwen对话格式
            messages = [
                {"role": "user", "content": item['input']},
                {"role": "assistant", "content": item['output']}
            ]
            
            # 应用chat template并编码
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # 编码并截断
            inputs = tokenizer(
                text, 
                max_length=seq_len, 
                truncation=True, 
                return_tensors="pt"
            )
            
            calibration_dataset.append({
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0]
            })
        except Exception as e:
            skipped += 1
            continue
    
    if skipped > 0:
        print(f"  ⚠️ 跳过 {skipped} 条无效数据")
    
    print(f"✅ 校准数据处理完成，共 {len(calibration_dataset)} 条有效数据")
    return calibration_dataset


def main():
    """主量化流程"""
    print("✨ 开始配置量化参数...")
    
    # ======================
    # 用户自定义配置区
    # ======================
    
    # 基础模型
    BASE_MODEL_NAME = 'Qwen3-0.6B'
    
    # 输出目录配置
    OUTPUT_BASE_DIR = "../../output"
    
    # 模型配置
    MERGED_MODEL_PATH = f"{OUTPUT_BASE_DIR}/merged_model/{BASE_MODEL_NAME}/lora_merged_model"
    
    # 数据集配置
    CALIBRATION_DATASET = "../../datasets/peft_data.jsonl"
    
    # 量化路径
    QUANTIZED_MODEL_DIR = f"{OUTPUT_BASE_DIR}/quantized_model/{BASE_MODEL_NAME}-gptq-4bit"
    
    # ======================
    # 量化参数配置区
    # ======================
    
    # 量化配置
    QUANTIZE_CONFIG = QuantizeConfig(
        bits=4,                  # 4-bit量化
        group_size=128,          # 与Qwen3的head_dim匹配
        damp_percent=0.01,       # 阻尼百分比
        desc_act=False,          # 禁用desc_act以提高速度
        static_groups=False,     # 不使用静态组
        sym=True,                # 对称量化
        true_sequential=True,    # 顺序量化
    )
    
    # 校准配置
    CALIBRATION_CONFIG = {
        "n_samples": 100,       # 校准样本数量
        "seq_len": 1024,        # 序列长度
    }
    
    # ======================
    # 路径配置结束
    # ======================
    
    print("\n📌 配置详情:")
    print(f"  • 待量化模型路径: {MERGED_MODEL_PATH}")
    print(f"  • 校准数据集路径: {CALIBRATION_DATASET}")
    print(f"  • 量化后模型保存路径: {QUANTIZED_MODEL_DIR}")
    print(f"  • 量化参数: bits={QUANTIZE_CONFIG.bits}, group_size={QUANTIZE_CONFIG.group_size}")
    print(f"  • 校准参数: 样本数={CALIBRATION_CONFIG['n_samples']}, 序列长度={CALIBRATION_CONFIG['seq_len']}")
    
    # 验证模型路径
    if not os.path.exists(MERGED_MODEL_PATH):
        raise FileNotFoundError(f"模型路径不存在: {MERGED_MODEL_PATH}")
    
    # 加载tokenizer
    print("\n🔄 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MERGED_MODEL_PATH, 
        use_fast=True, 
        trust_remote_code=True
    )
    
    # 如果没有pad_token，使用eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载并处理校准数据
    print("\n📊 准备校准数据...")
    calibration_dataset = load_calibration_data(
        CALIBRATION_DATASET,
        tokenizer,
        n_samples=CALIBRATION_CONFIG["n_samples"],
        seq_len=CALIBRATION_CONFIG["seq_len"]
    )
    
    if len(calibration_dataset) == 0:
        raise ValueError("校准数据集为空，请检查数据格式和路径")
    
    # 加载模型
    print("\n🔧 加载模型...")
    model = GPTQModel.from_pretrained(
        MERGED_MODEL_PATH,
        quantize_config=QUANTIZE_CONFIG,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # 执行量化
    print("\n⚡ 执行量化过程...")
    model.quantize(calibration_dataset)
    
    # 保存量化模型
    print("\n💾 保存量化模型...")
    os.makedirs(QUANTIZED_MODEL_DIR, exist_ok=True)
    model.save_quantized(QUANTIZED_MODEL_DIR)
    tokenizer.save_pretrained(QUANTIZED_MODEL_DIR)
    
    print(f"\n🎉 量化完成! 模型已保存至: {QUANTIZED_MODEL_DIR}")
    
if __name__ == "__main__":
    main()