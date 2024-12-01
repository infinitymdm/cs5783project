#! /usr/bin/env bash

pushd latent-diffusion
CUDA_VISIBLE_DEVICES=0 python3 scripts/sample_diffusion.py -r /opt/ckpts/2024-11-30T20-00-24_txt2img-mini/ -n 4 --batch_size 1 -c 20 -e 0.0
popd
