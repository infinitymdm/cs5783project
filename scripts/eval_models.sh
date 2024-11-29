#!/bin/bash
CKPT_LOCATION=/opt/ckpts
OUTPUTFILE=./outfile

declare -a arr=($(ls $CKPT_LOCATION | xargs))
print arr
pushd latent-diffusion
for i in "${arr[@]}"
do
    model="$CKPT_LOCATION/$i/model.ckpt"
    echo $model
    loss=$(CUDA_VISIBLE_DEVICES=0 python3 main.py -r "$CKPT_LOCATION/$i/model.ckpt" -b ../configs/txt2img-mini.yaml | grep "/loss\s" | xargs | cut -d " " -f 2)
    echo "model_$i : $loss" >> $OUTPUTFILE
done
popd
