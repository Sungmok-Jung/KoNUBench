# KoNUBench
Developing KoNUBench, a benchmark for evaluating sentence-level negation understanding ability of LLM in Korean

## Install & SetUp
~~~bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ~

git clone https://github.com/Sungmok-Jung/KoNUBench.git
~~~

### Add Tasks to lm-eval-harness
To use KoNUBench tasks inside lm-eval-harness, copy the task directories into your lm-eval-harness installation:
~~~bash
cp -r KoNUBench/dataset/{kmmlu_neg,kmmlu_pos,kobest_boolq_neg,ko_nubench_cloze,ko_nubench_symbol} \
      /root/sky_workdir/lm-evaluation-harness/lm_eval/tasks/
~~~

### Evaluation
To evaluate model performance on KoNUBench, use the following command:
~~~bash
lm_eval \
    --model hf \
    --model_args pretrained={Model_Name},trust_remote_code=True \
    --tasks ko_nubench_symbol,ko_nubench_cloze \
    --output_path {Directory_to_store_results} \
    --log_samples \
    --batch_size auto
~~~

### Supervised-Fine-Tuning
KoNUBench also provides training data for fine-tuning models on negation understanding. Below is an example using torchrun with distributed training:
~~~bash
  torchrun \
  --nnodes=$SKYPILOT_NUM_NODES \
  --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --node_rank=${SKYPILOT_NODE_RANK} \
  --master_port=8008 \
  /root/KoNUBench/sft/main.py \
    --model {Model_Name} \
    --global_batch 128 \
    --micro_batch 16 \
    --learning_rate 3e-5 \
    --seed 1200 \
    --deepspeed_stage 1 \
    --dtype bf16 \
    --num_cpus 16 \
    --checkpoint_interval 200 \
    --checkpoint_path /mnt/sm/ckp \
    --wandb_off \
    --epochs 3
~~~
