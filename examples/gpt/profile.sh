#!/bin/bash


# Append input_length and output_length to the nsys-rep file name
output_file="profiled_result/gpt_analysis"

# Run the nsys profile command
nsys profile -t cuda,nvtx -o ${output_file} -f true --capture-range=cudaProfilerApi --capture-range-end=repeat \
    python3 ../run.py --engine_dir gpt_6.7b/trt_engines/fp16/1-gpu \
    --tokenizer_dir gpt2 \
    --max_input_length 128 \
    --max_output_len 128 \
    --warmup_iter 30 \
    --profile_iter 100 \
    --run_profiling

for input_length in {1..128}; do
    for output_length in {1..128}; do
        # Generate an individual CSV file for each .nsys-rep file
        index=$(((input_length-1)*128 + output_length)) 
        nsys stats --format=csv --force-export=true ${output_file}.${index}.nsys-rep > ${output_file}_i${input_length}_o${output_length}.csv
        rm ${output_file}.${index}.nsys-rep
        rm ${output_file}.${index}.sqlite
    done
done
