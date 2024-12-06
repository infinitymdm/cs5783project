pushd latent-diffusion
CUDA_VISIBLE_DEVICES=0 python3 main.py -r "$1" -b "$1/config.yaml"
popd
