#! /usr/bin/env bash

# Run this script from the main repo directory.
REPO_DIR=$(pwd)
CKPT_LOCATION=~/ckpts
OUTPUTFILE=./outfile
LOGDIR="$REPO_DIR/logs"

if [ -e $LOGDIR ]; then
    rm -rf $LOGDIR
fi

mkdir $LOGDIR
echo $LOGDIR

#declare -a arr=($(ls $CKPT_LOCATION | xargs))
#print arr
pushd latent-diffusion
for i in $(ls $CKPT_LOCATION | grep "^2024")
do
    model="$CKPT_LOCATION/$i/model.ckpt"
    echo $model
    config="$(echo "$i" | tr "_" "\n" | tail -n 1)"
    echo $config
    CUDA_VISIBLE_DEVICES=0 python3 main.py -r "$model" -l "$LOGDIR/$i" -b "../configs/$config".yaml
    #echo "model_$i : $loss" >> $OUTPUTFILE
done
popd
