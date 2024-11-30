#!/bin/bash
# Run this script from the main repo directory.
REPO_DIR=$(pwd)
CKPT_LOCATION=/opt/ckpts
OUTPUTFILE=./outfile
LOGDIR=$REPODIR/logs

if [ -e $LOGDIR ]; then
    rm -rf $LOGDIR
fi

mkdir $LOGDIR

declare -a arr=($(ls $CKPT_LOCATION | xargs))
print arr
pushd latent-diffusion
for i in "${arr[@]}"
do
    model="$CKPT_LOCATION/$i/model.ckpt"
    echo $model
    loss=$(CUDA_VISIBLE_DEVICES=0 python3 main.py -n "$i" -r "$CKPT_LOCATION/$i/model.ckpt" -l $LOGDIR -b ../configs/txt2img-mini.yaml)
    echo "model_$i : $loss" >> $OUTPUTFILE
done
popd
