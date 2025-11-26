import argparse
import os
from glob import glob
import subprocess
import time as t
from tqdm import tqdm
import psutil

import torch
import torch.distributed as dist
import torchinfo
import deepspeed
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import datasets
import json
import wandb

def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def load_model_tokenizer(args, summary=False):
    model_loading_start = t.time()
    
    # model
    model = AutoModelForCausalLM.from_pretrained(args.model)

    if args.activation_recomputation:
        model.gradient_checkpointing_enable()
    
    if summary:
        torchinfo.summary(model)
    
    model_loading_end = t.time()
    model_loading_duration = round(model_loading_end - model_loading_start, 2)
    args.debug_model_loading_duration = model_loading_duration

    # tokenizer
    tok_loading_start = t.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        # handle special cases
        if  "qwen" in args.model:
            # Qwen's trust_remote_code tokenizer does not allow for adding special tokens
            tokenizer.pad_token = "<|endoftext|>"
        elif (
            tokenizer.__class__.__name__ == "RWKVWorldTokenizer"
            or tokenizer.__class__.__name__ == "Rwkv5Tokenizer"
        ):
            assert tokenizer.pad_token_id == 0
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tok_loading_end = t.time()
    tok_loading_duration = round(tok_loading_end - tok_loading_start, 2)
    args.debug_tokenizer_loading_duration = tok_loading_duration    

    return model, tokenizer


def sft_load_dataset(args):
    # TODO : args.dataset_names 관련 처리
    assert args.dataset_names is None, f"args.dataset_names is not None : {args.dataset_names}"
    
    # TODO : 여러개 jsonl 파일에 대한 처리 - args 수정만 하면 됨
    json_paths = glob(f"{args.dataset_root}/tokenized/sft/*/*.jsonl")
    # json_paths = ...     # f"{args.dataset_root}/tokenized/sft/{args.dataset_names}"
    # json_paths = ["/home/s1/jongmin/0_chatbot/datasets/tokenized/sft/121.한국어_성능이_개선된_초거대AI_언어모델_개발_및_데이터/SFTlabel.jsonl"]
    if not json_paths:
        raise FileNotFoundError(f"No files found in '{args.dataset_root}/tokenized/sft/'")
    
    features = datasets.Features({
        "id": datasets.Value("string"),  # ← 포인트: 문자열로 통일
        "input_ids": datasets.Sequence(datasets.Value("int64")),
        "target": datasets.Sequence(datasets.Value("int64")),
    })
    
    parts = []
    bad_files = []
    for path in json_paths:
        try:
            ds = datasets.load_dataset(
                "json",
                data_files=path,
                split="train",
                features=features,
            )
            if len(ds) == 0:
                print(f"[WARNING] Empty dataset: '{path}'")
            parts.append(ds)
        except Exception as e:
            bad_files.append((path, repr(e)))
            print(f"[ERROR] Failed to load {path}: {e}")
    
    if bad_files:
        msg = "\n".join([f"- {p}: {err}" for p, err in bad_files[:10]])
        raise RuntimeError(
            f"{len(bad_files)} file(s) failed to load.\n"
            f"Examples:\n{msg}\n"
            f"(showing up to 10; fix or exclude these files and retry)"
        )

    if not parts:
        raise RuntimeError("All files were empty or failed to load.")

    train_dataset = datasets.concatenate_datasets(parts)
    train_dataset = train_dataset.shuffle(seed=args.seed)
    print(f"Total rows: {len(train_dataset)}")

    args.num_dataset_rows = len(train_dataset)
    return train_dataset

def tokenize_load_dataset(args, tokenizer: AutoTokenizer):
    prompt = f"문제: 다음의 원문을 standard negation을 활용하여 올바르게 부정하시오.\n\nstandard negation:\n- 주절의 서술어를 한국어의 부정 표현을 활용해 부정함으로써 원문 P를  ¬P로 만든다.\n- 주절의 서술어 외의 나머지 부분은 수정하지 않는다.\n- 원문이 조건문일 때는 논리적 규칙(¬(P → Q) ≡ P ∧ ¬Q)을 따라 부정한다.\n- 주절이 여러 개일 경우 드모르간의 법칙(예. ¬(P ∧ Q) ≡ ¬P ∨ ¬Q)에 따라 모든 주절의 서술어를 부정한다.\n\n한국어의 부정 표현:\n- 안 계열: 안, -지 않다\n- 못 계열: 못, -지 못하다\n- 어휘적 부정: 상보 반의어를 활용한 부정(이다/아니다, 있다/없다, 참석하다/불참하다 등)\n\n원문:"
    train_dataset = []
    with open(file=args.dataset_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    for d in data:
        ctx = f'{prompt} {d["original_sentence"]}\n부정문:'

        ctx_tokenized = tokenizer(ctx, add_special_tokens=True)
        ctx_ids = ctx_tokenized["input_ids"]

        end = f' {d["standard_negation"]}'
        end_tokenized = tokenizer(end, add_special_tokens=False)
        end_ids = end_tokenized["input_ids"]

        seq = ctx_ids + end_ids
        target = ([-100] * len(ctx_ids)) + end_ids


        train_dataset.append({
            "context": d["original_sentence"],
            "ending": d["standard_negation"],
            "input_ids": seq,
            "target": target
        })

    return train_dataset


def wandb_init(args):
    wandb_name = f"{args.model}/lr{args.learning_rate}/seed{args.seed}"
    checkpoint_path = f"{args.checkpoint_path}/{wandb_name}"
    if not args.wandb_off and args.global_rank == 0:
        wandb.login()
        if not os.path.exists(f"{checkpoint_path}/wandb") :
            os.makedirs(f"{checkpoint_path}/wandb")
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=wandb_name,
            config=args,
            save_code=True,
            dir=checkpoint_path
        )
        

def deepspeed_init(args):
    deepspeed.init_distributed(verbose=False)
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.global_rank = torch.distributed.get_rank()
    args.local_rank = int(os.getenv("LOCAL_RANK", "0")) 
    
    
def deepspeed_destroy():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    
def load_deepspeed_config(args):
    loading_start = t.time()
    
    # dataset size (after distributed init)
    grad_accum = args.global_batch // args.micro_batch // args.world_size
    assert args.global_batch == grad_accum * args.micro_batch * args.world_size, f"grad_accum: {grad_accum}, global_batch: {args.global_batch}, micro_batch: {args.micro_batch}, world_size: {args.world_size}"
    
    steps_per_epoch = args.num_dataset_rows // args.global_batch
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps*args.warmup_ratio)
    
    # (save to args)
    args.grad_accum = grad_accum
    args.steps_per_epoch = steps_per_epoch
    args.total_steps = total_steps
    
    ds_config = {
        "zero_optimization":{
            "stage":args.deepspeed_stage,
        },
        "train_batch_size": args.global_batch,
        "train_micro_batch_size_per_gpu": args.micro_batch,
        "gradient_accumulation_steps":grad_accum, # added
        "fp16": {
            "enabled": True if args.dtype=="fp16" else False,
            "loss_scale": 0,  # 동적 스케일링 활성화
            "initial_scale_power": 16,  # 초기 스케일 2^16
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16":{
            "enabled": True if args.dtype=="bf16" else False,
        },
        "optimizer": {
            "type": "AdamW", 
                "params": {
                    "lr": args.learning_rate,
                    "betas": [0.9, 0.999], # 0.9, 0.999
                    "eps":1e-8, # 1e-8
                    "weight_decay": 0.01,
                    # "max_grad_norm ":1.0, # added
            }
        },
        "gradient_clipping": 1.0,
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps":total_steps,
                # "warmup_min_ratio": 0.1,
                "warmup_num_steps": warmup_steps,
                "warmup_type":"linear", # log -> linear
            }
        },
        "activation_checkpointing": {
            "partition_activations": True if args.activation_recomputation else False,
            "cpu_checkpointing": True if args.activation_recomputation else False,
            "contiguous_memory_optimization": True if args.activation_recomputation else False,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        }
    }
    return ds_config


def load_checkpoint(args, model_engine):
    raise NotImplementedError("load_checkpoint is not implemented yet.")
