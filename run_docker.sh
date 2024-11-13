#!/bin/bash

docker run --runtime=nvidia --gpus '"device=4,5,6,7"' --shm-size=8g -it -v $PWD:/workspace -w /workspace tensorrt_llm/release:latest