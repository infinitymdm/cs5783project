pushd latent-diffusion
CUDA_VISIBLE_DEVICES=0 python3 main.py -r ../uniform_soup.pt -b ../configs/txt2img-mini.yaml
popd
