#!/bin/bash

python3 ../generate_checkpoint_config.py --architecture GPTForCausalLM \
        --vocab_size 51200 \
        --hidden_size 4096 \
        --num_hidden_layers 32 \
        --num_attention_heads 32 \
        --dtype float16 \
        --tp_size 1 \
        --output_path gpt_6.7b/trt_ckpt/fp16/1-gpu/config.json

trtllm-build --model_config gpt_6.7b/trt_ckpt/fp16/1-gpu/config.json \
        --gemm_plugin auto \
        --max_batch_size 1 \
        --max_input_len 1024 \
        --max_seq_len 2048 \
        --output_dir gpt_6.7b/trt_engines/fp16/1-gpu \
        --workers 8