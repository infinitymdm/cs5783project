#1 /usr/bin/env bash

echo "Training model with config $1 ..."
rm ~/.cache/autoencoders/data/*/.ready
pushd latent-diffusion
CUDA_VISIBLE_DEVICES=0; python3 main.py --base "$1" -t --gpus 1, --no-test
popd
