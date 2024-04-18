model_size='125m'
n_layer=12
n_head=12
n_embd=768
n_gpu=1
tp_size=1


python3 ../generate_checkpoint_config.py --architecture GPTForCausalLM \
        --vocab_size 51200 \
        --hidden_size "$n_embd" \
        --num_hidden_layers "$n_layer" \
        --num_attention_heads "$n_head" \
        --dtype float16 \
        --tp_size "$tp_size" \
        --output_path gpt_"$model_size"/trt_ckpt/fp16/"$n_gpu"-gpu/config.json
echo generation of config done
trtllm-build --model_config gpt_"$model_size"/trt_ckpt/fp16/"$n_gpu"-gpu/config.json \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --context_fmha enable \
        --gemm_plugin float16 \
        --max_batch_size 256 \
        --output_dir gpt_"$model_size"/trt_engines/fp16/"$n_gpu"-gpu \
        --workers "$n_gpu" \
        --use_custom_all_reduce disable
