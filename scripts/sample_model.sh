#! /usr/bin/env bash

pushd latent-diffusion
CUDA_VISIBLE_DEVICES=0 python3 scripts/sample_diffusion.py -r "$1" -n 4 -v --batch_size 1
popd
