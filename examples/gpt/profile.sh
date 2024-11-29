#!/bin/bash


# Append input_length and output_length to the nsys-rep file name
output_file="profiled_result/gpt3_6.7b_analysis"


# generate traces with diverse KV cache
for input_length in {1..1}; do
    for output_length in {1..1024}; do
        # Generate an individual CSV file for each .nsys-rep file
        # Run the nsys profile command
        nsys profile -t cuda,nvtx -o ${output_file} -f true --capture-range=cudaProfilerApi --capture-range-end=repeat \
            python3 ../run.py --engine_dir gpt_6.7b/trt_engines/fp16/1-gpu \
            --tokenizer_dir gpt2 \
            --max_input_length ${input_length} \
            --max_output_len ${output_length} \
            --warmup_iter 10 \
            --profile_iter 1 \
            --run_profiling

        index=$(((input_length-1)*2 + output_length)) 
        nsys stats --format=csv --force-export=true ${output_file}.nsys-rep > ${output_file}_i${input_length}_o${output_length}.csv
        rm ${output_file}.nsys-rep
        rm ${output_file}.sqlite
    done
done

# generate traces with diverse input size
for input_length in {1..1024}; do
    for output_length in {1..1}; do
        # Generate an individual CSV file for each .nsys-rep file
        # Run the nsys profile command
        nsys profile -t cuda,nvtx -o ${output_file} -f true --capture-range=cudaProfilerApi --capture-range-end=repeat \
            python3 ../run.py --engine_dir gpt_6.7b/trt_engines/fp16/1-gpu \
            --tokenizer_dir gpt2 \
            --max_input_length ${input_length} \
            --max_output_len ${output_length} \
            --warmup_iter 10 \
            --profile_iter 1 \
            --run_profiling

        index=$(((input_length-1)*2 + output_length)) 
        nsys stats --format=csv --force-export=true ${output_file}.nsys-rep > ${output_file}_i${input_length}_o${output_length}.csv
        rm ${output_file}.nsys-rep
        rm ${output_file}.sqlite
    done
done
