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

def main(args):
    print(args.config)

    config = OmegaConf.load(args.config)
    #print(config)
    base_model = instantiate_from_config(config.model)

    print(base_model["state_dict"])

    
    
    

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
