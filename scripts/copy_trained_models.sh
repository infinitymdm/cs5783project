#! /usr/bin/env bash

for f in `ls latent-diffusion/logs`
do
    mkdir /opt/ckpts/$f
    mv latent-diffusion/logs/$f/testtube /opt/ckpts/$f/.
    mv latent-diffusion/logs/$f/configs/*-project.yaml /opt/ckpts/$f/config.yaml
    mv latent-diffusion/logs/$f/checkpoints/epoch*.ckpt /opt/ckpts/$f/model.ckpt
done
