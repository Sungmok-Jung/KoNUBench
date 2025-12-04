import argparse
import os, glob
import numpy as np
import random
import math
import torch
import deepspeed


from load import (
    load_model_tokenizer,
    tokenize_load_dataset,
    sft_load_dataset,
    deepspeed_init,
    deepspeed_destroy,
    load_deepspeed_config,
    load_checkpoint
)
from train import train
from utils import wandb_init

def parse_args():
    parser = argparse.ArgumentParser()
    
    # model architecture
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B", type=str)
    parser.add_argument("--model_seq_len", default=2048, type=int)
    
    # dataset
    parser.add_argument("--dataset_path", default="/root/KoNUBench/dataset/ko_nubench/train/train.json")
    
    # training settings
    parser.add_argument("--global_batch", default=512, type=int)
    parser.add_argument("--micro_batch", default=1, type=int)
    parser.add_argument("--learning_rate", default=4e-4, type=float)    
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--epochs", default=1, type=int)
    
    # (misc.)
    parser.add_argument("--seed", action="store", default=308, type=int)
    parser.add_argument("--deepspeed_stage", action="store", default=0, type=int)
    parser.add_argument("--dtype", action="store", default="bf16")
    parser.add_argument("--num_cpus", action="store", default=16, type=int)
    parser.add_argument("--activation_recomputation", action="store_true")
    parser.add_argument("--cache_root", default="/shared/erc/lab08/jongmin/.cache/chatbot")
    
    # logging / checkpointing
    parser.add_argument("--wandb_off", action="store_true")
    parser.add_argument("--wandb_entity", default="jongmin-dev-seoul-national-university")
    parser.add_argument("--wandb_project_name", default="chatbot")
    parser.add_argument("--checkpoint_path", default="/mnt/sm/KoNUBench/sft/models")
    parser.add_argument("--checkpoint_interval", default=200, type=int)
    parser.add_argument("--prev_ckpt")
    
    args = parser.parse_args()

    args_dict = vars(args)
    print()
    print("[==================== Arguments ====================]")
    for key, value in args_dict.items():
        print(f"{key}: {value}")
    print("[===================================================]\n")

    return args

def main():
        
    args = parse_args()

    

    # load tokenizer & model
    model, tokenizer = load_model_tokenizer(args=args, summary=True)
    model.to("cuda") # potential problem
    
    train_dataset = tokenize_load_dataset(args, tokenizer)
    args.num_dataset_rows = len(train_dataset)

    # deepspeed init
    deepspeed_init(args=args)
    ds_config = load_deepspeed_config(args=args)
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, 
        model_parameters=[p for p in model.parameters() if p.requires_grad == True], 
        config=ds_config,
    )
    model_engine = model_engine.to("cuda") # potential problem
    
    # wandb initialize
    wandb_init(args)
        
    # train
    train(
        args=args,
        tokenizer=tokenizer,
        model_engine=model_engine,
        dataset=train_dataset
    )
    
    deepspeed_destroy()
    return


if __name__ == "__main__":
    main()