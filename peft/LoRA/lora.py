#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Qwen3-0.6B LoRA 微调训练脚本
功能特点：
1. 支持完全自定义路径配置
2. 同时保存LoRA适配器和合并后的完整模型
3. 清晰的路径管理
"""

import os
import torch
from modelscope import snapshot_download
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, TaskType, get_peft_model
from swanlab.integration.transformers import SwanLabCallback


def load_and_preprocess_data(dataset_path, tokenizer):
    """加载数据集并进行预处理"""
    print(f"🔄 加载数据集: {dataset_path}...")
    
    # 加载 JSONL 格式数据集
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # 定义预处理函数
    def preprocess_function(samples):
        # 拼接输入输出并添加 EOS 标记
        texts = [
            inp + out + tokenizer.eos_token 
            for inp, out in zip(samples["input"], samples["output"])
        ]
        
        # 编码文本
        tokenized = tokenizer(
            texts,
            padding="longest",
            max_length=12000,
            truncation=True,
            truncation_side="right",
            return_tensors="pt",
            return_attention_mask=True
        )
        
        # 创建标签（仅计算输出部分的损失）
        input_lens = [
            len(tokenizer.encode(inp, add_special_tokens=False)) 
            for inp in samples["input"]
        ]
        labels = tokenized["input_ids"].clone()
        
        for i, input_len in enumerate(input_lens):
            # 忽略输入部分的损失计算
            labels[i, :input_len] = -100
            
            # 确保 EOS 标记参与损失计算
            if tokenized["input_ids"][i, -1] == tokenizer.eos_token_id:
                labels[i, -1] = tokenizer.eos_token_id
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }
    
    # 应用预处理
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=8,
        remove_columns=["input", "output"],
        num_proc=4  # 使用多进程加速
    )
    
    print(f"✅ 数据集处理完成，样本数: {len(tokenized_dataset)}")
    return tokenized_dataset


def setup_lora_model(base_model_path):
    """配置 LoRA 微调参数"""
    print(f"🔧 配置 LoRA 微调模型: {base_model_path}...")
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.enable_input_require_grads()  # 梯度检查点必需
    
    # 配置 LoRA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    
    # 应用 LoRA
    model = get_peft_model(model, config)
    print("✅ LoRA 模型配置完成")
    return model


def setup_training(model, tokenizer, tokenized_dataset, output_dir):
    """设置训练参数并初始化训练器"""
    print(f"⚙️ 配置训练参数 (输出目录: {output_dir})...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 训练参数
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=5,
        num_train_epochs=5,
        save_steps=50,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
        fp16=True,  # 启用半精度训练
        optim="adamw_torch_fused",  # 使用优化版优化器
        ddp_find_unused_parameters=False
    )
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
    
    # 初始化 SwanLab 监控
    swanlab_callback = SwanLabCallback(
        project="Qwen3-0.6B",
        experiment_name="Qwen3-0.6B-Lora"
    )
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[swanlab_callback]
    )
    print("✅ 训练器初始化完成")
    return trainer


def main():
    """主训练流程 - 所有路径配置在此处自定义"""
    print("✨ 开始配置训练参数...")
    
    # ======================
    # 用户自定义路径配置区
    # ======================
    
    # 基础模型配置
    BASE_MODEL_SOURCE = "Qwen/Qwen3-0.6B"  # ModelScope 模型标识或本地路径
    BASE_MODEL_NAME = BASE_MODEL_SOURCE.split('/')[-1]
    MODEL_CACHE_DIR = "/root/autodl-tmp/"   # 模型缓存目录
    
    # 数据集配置
    DATASET_PATH = "../../datasets/peft_data.jsonl"
    
    # 输出目录配置
    OUTPUT_PATH = "../../output"
    TRAINING_OUTPUT_DIR = f"{OUTPUT_PATH}/training_checkpoints/{BASE_MODEL_NAME}/lora_checkpoint"  # 训练检查点目录
    LORA_ADAPTER_DIR = f"{OUTPUT_PATH}/peft_model/{BASE_MODEL_NAME}/lora_adapter"               # LoRA适配器保存目录
    MERGED_MODEL_DIR = f"{OUTPUT_PATH}/merged_model/{BASE_MODEL_NAME}/lora_merged_model"               # 合并模型保存目录
    
    # ======================
    # 路径配置结束
    # ======================
    
    print("📌 自定义路径配置:")
    print(f"  • 基础模型来源: {BASE_MODEL_SOURCE}")
    print(f"  • 模型缓存目录: {MODEL_CACHE_DIR}")
    print(f"  • 数据集路径: {DATASET_PATH}")
    print(f"  • 训练检查点目录: {TRAINING_OUTPUT_DIR}")
    print(f"  • LoRA适配器保存目录: {LORA_ADAPTER_DIR}")
    print(f"  • 合并模型保存目录: {MERGED_MODEL_DIR}")
    
    # 步骤1: 获取基础模型路径（自动处理下载或使用本地路径）
    print("\n🔍 确定基础模型路径...")
    if os.path.isdir(BASE_MODEL_SOURCE):
        print(f"  • 检测到本地模型路径: {BASE_MODEL_SOURCE}")
        base_model_path = BASE_MODEL_SOURCE
    else:
        print(f"  • 从 ModelScope 下载模型: {BASE_MODEL_SOURCE}")
        base_model_path = snapshot_download(
            BASE_MODEL_SOURCE,
            cache_dir=MODEL_CACHE_DIR,
            revision="master"
        )
        print(f"  • 模型已下载至: {base_model_path}")
    
    # 步骤2: 加载 tokenizer 和数据集
    print("\n🔄 加载 tokenizer 和数据集...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, 
        use_fast=False, 
        trust_remote_code=True
    )
    
    tokenized_dataset = load_and_preprocess_data(DATASET_PATH, tokenizer)
    
    # 步骤3: 配置 LoRA 模型
    print("\n🔧 配置 LoRA 微调模型...")
    model = setup_lora_model(base_model_path)
    
    # 步骤4: 配置训练
    print("\n⚙️ 配置训练参数...")
    trainer = setup_training(model, tokenizer, tokenized_dataset, TRAINING_OUTPUT_DIR)
    
    # 步骤5: 启动训练
    print("\n🔥 开始训练...")
    trainer.train()
    
    # 步骤6: 保存结果
    print("\n💾 保存训练结果...")
    
    # 6.1 保存 LoRA 适配器 (仅微调参数)
    print(f"  • 保存 LoRA 适配器至: {LORA_ADAPTER_DIR}")
    os.makedirs(LORA_ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(LORA_ADAPTER_DIR)
    tokenizer.save_pretrained(LORA_ADAPTER_DIR)
    print(f"    ✅ LoRA 适配器已保存 (约 10-20MB)")
    print(f"    💡 提示: 此目录仅包含微调参数，推理时需与基础模型配合使用")
    
    # 6.2 保存合并后的完整模型
    print(f"  • 合并并保存完整模型至: {MERGED_MODEL_DIR}")
    os.makedirs(MERGED_MODEL_DIR, exist_ok=True)
    
    # 合并 LoRA 适配器到基础模型
    merged_model = model.merge_and_unload()
    
    # 保存完整模型
    merged_model.save_pretrained(MERGED_MODEL_DIR)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)
    print(f"    ✅ 合并后的完整模型已保存")
    print(f"    💡 提示: 此目录包含完整模型，可直接用于推理部署")
    
    print("\n🎉 训练和保存流程全部完成!")


if __name__ == "__main__":
    main()