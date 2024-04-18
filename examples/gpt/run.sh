model_size='125m'
n_gpu=1

python3 ../generate_input.py

mpirun -np "$n_gpu" --allow-run-as-root \
    python3 ../run.py --engine_dir gpt_"$model_size"/trt_engines/fp16/"$n_gpu"-gpu \
        --max_output_len 1 \
        --run_profiling \
        --input_file ../input.txt