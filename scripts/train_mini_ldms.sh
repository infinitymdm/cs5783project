#! /usr/bin/env bash

for i in {1..2}
do
    echo "Training model $i ..."
    rm ~/.cache/autoencoders/data/*/.ready
    pushd latent-diffusion
    CUDA_VISIBLE_DEVICES=0 python3 main.py --base ../configs/config"$i".yaml -t --gpus 1, --no-test
    popd
done
