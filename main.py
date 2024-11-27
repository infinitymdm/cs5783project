#!/usr/bin/python3

import argparse, os, sys, datetime, glob, importlib, csv
import wget
import torch
import torchvision
import pytorch_lightning as pl
import clip
import os
import json
import operator
from pathlib import Path
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#from ..."model-soups" import utils as msutils

import importlib
msutils = importlib.import_module("model-soups.utils")
#from model_soups.utils import get_model_from_sd, test_model_on_dataset
#from msutils import get_model_from_sd

get_model_from_sd = msutils.get_model_from_sd

#import importlib

#instantiate_from_config = importlib.import_module("instantiate_from_config", "..latent-diffusion.ldm.util")

from ldm.util import instantiate_from_config

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('~/models'),
        help="Where to download the models.",
    )
    # Don't need this... yet
    #parser.add_argument(
    #    "--download-models", action="store_true", default=False,
    #)

    parser.add_argument(
        "--eval-individual-models", action="store_true", default=False,
    )

    parser.add_argument(
        "--uniform-soup", action="store_true", default=False,
    )

    parser.add_argument(
        "--greedy-soup", action="store_true", default=False,
    )

    parser.add_argument(
        "--plot", action="store_true", default=False,
    )

    parser.add_argument(
        "--config", type=str, default=os.path.relpath('configs/txt2img-mini.yaml'),
        help="Primary config."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    return parser.parse_args()

def get_model_paths(models_dir):
    return [Path(c) for c in models_dir.glob('**/*.ckpt')]

def main(args):
    # Define model path
    models_dir = Path(args.model_location)
    # Load the LDM config.
    config = OmegaConf.load(args.config)

    print(msutils)

    # Instantiate the LDM model. This model is a Pytorch Lightning model
    base_model = instantiate_from_config(config.model)

    if args.uniform_soup:
        model_paths = get_model_paths(models_dir)
        NUM_MODELS = len(model_paths)
        #base_model = torch.load(model_paths[0], map_location="cpu")
        for j, model_path in enumerate(model_paths):
            model_dict = torch.load(model_path, map_location="cpu")["state_dict"]
            #print(model_dict.keys())

            assert os.path.exists(model_path)
            if j == 0:
                uniform_soup = {k : v * (1./NUM_MODELS) for k, v in model_dict.items()}
            else:
                uniform_soup = {k : v * (1./NUM_MODELS) + uniform_soup[k] for k, v in model_dict.items()}

        model = instantiate_from_config(config.model)
        model.load_state_dict(uniform_soup)

        torch.save(uniform_soup, "uniform_soup.pt")
        
        

    
    
    

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
