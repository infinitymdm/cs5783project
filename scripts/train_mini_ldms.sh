#! /usr/bin/env bash

for i in {1..50}
do
    echo "Training model $i ..."
    pushd latent-diffusion
    CUDA_VISIBLE_DEVICES=0 python3 main.py --base ../configs/txt2img-mini.yaml -t --gpus 0, --no-test
    popd
done
