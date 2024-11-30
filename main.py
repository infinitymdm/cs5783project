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

import pathlib
from pathlib import Path
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import shutil

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
        "-m",
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
        "-l",
        "--logdir",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="Log directory for individual models",
    )
    
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
        "-r",
        "--results",
        type=str,
        default=os.path.relpath('./results'),
        help="Directory for storing intermediate greedy soup results",
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

def get_eval_paths(eval_dir):
    return [Path(c) for c in eval_dir.glob('**/**/**/*.csv')]

def get_ckpt(models_dir, dir_name):
    path_to_model = Path(os.path.join(models_dir, dir_name))
    return next(path_to_model.glob('**/*.ckpt'),None)

def get_csv(models_dir, dir_name):
    path_to_model = Path(os.path.join(models_dir, dir_name))
    return next(path_to_model.glob('**/**/**/*.csv'), None)

def main(args):
    INDIVIDUAL_MODEL_RESULTS_FILE = 'individual_model_results.jsonl'
    UNIFORM_SOUP_RESULTS_FILE = 'uniform_soup_results.jsonl'
    GREEDY_SOUP_RESULTS_FILE = 'greedy_soup_results.jsonl'
    # Define model path
    models_dir = Path(args.model_location)
    logdir = Path(args.logdir) if args.logdir else None
    print(logdir)
    # Load the LDM config.
    config = OmegaConf.load(args.config)

    print(msutils)

    # Instantiate the LDM model. This model is a Pytorch Lightning model
    base_model = instantiate_from_config(config.model)
    model_paths = get_model_paths(models_dir)
    NUM_MODELS = len(model_paths)

    # Evlatuate Individual Models -------------------------------------
    if args.eval_individual_models:
        if os.path.exists(INDIVIDUAL_MODEL_RESULTS_FILE):
            os.remove(INDIVIDUAL_MODEL_RESULTS_FILE)

        if not logdir:
            raise exception("Need to have a log directory to create sorted list.")

        print("HERE")
        eval_paths = get_eval_paths(logdir)

        #eval_dirs = [x[0] for x in os.walk(logdir)]
        eval_dirs = next(os.walk(logdir))[1]

        print(eval_dirs)

        losses = []
        for j, eval_path in enumerate(eval_paths):
            #print(f"{j}, {eval_path}")
            df = pd.read_csv(eval_path)
            print(f"{eval_path}\n {df['val/loss'][0]}")
            losses.append(df['val/loss'][0])

        losses_list = dict(zip(eval_dirs, losses))
        print(losses_list)

        with open(INDIVIDUAL_MODEL_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(losses_list) + '\n')

    # Uniform soup ----------------------------------------------------
    if args.uniform_soup:
        #model_paths = get_model_paths(models_dir)
        #NUM_MODELS = len(model_paths)
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

        
    # Greedy soup -----------------------------------------------------
    if args.greedy_soup:

        results_dir=Path(args.results)
        # Reset greedy soup results file
        if os.path.exists(GREEDY_SOUP_RESULTS_FILE):
            os.remove(GREEDY_SOUP_RESULTS_FILE)

        # Reset results directory
        if os.path.exists(args.results):
            shutil.rmtree(Path(args.results))
        os.mkdir(args.results)

        # This series of functions reads the output JSON into a Pandas DataFrame
        # but it reads it in a different format than what it wrote the dict in.
        # I first convert the dataframe back into the correct dictionary form,
        # then conver the dictionary into a list of tuples, sorting by the second
        # element. I then recover the sorted diretory names.

        # If we had multiple datasets we validated on, this would have to be different.
        individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
        individual_model_val_accs = individual_model_db.to_dict(orient="records")[0]
        sorted_models = list(individual_model_val_accs.items())
        sorted_models.sort(key = lambda a : a[1])
        sorted_models = list(zip(*sorted_models))[0]

        individual_model_val_accs = sorted(individual_model_val_accs.items(), key=operator.itemgetter(1))
        # Start the soup by using the first ingredient.
        greedy_soup_ingredients = [sorted_models[0]]
        path_to_ckpt = get_ckpt(models_dir, sorted_models[0])
        greedy_soup_params = torch.load(path_to_ckpt)['state_dict']
        best_val_loss_so_far = individual_model_val_accs[0][1]
        #held_out_val_set = ImageNet2p(preprocess, args.data_location, args.batch_size, args.workers)

        for i in range(1, NUM_MODELS):
            print(f'Testing model {i} of {NUM_MODELS}')

            path_to_ckpt = get_ckpt(models_dir, sorted_models[i])
            new_ingredient_params = torch.load(path_to_ckpt)['state_dict']
            num_ingredients = len(greedy_soup_ingredients)
            potential_greedy_soup_params = {
                k : greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) +
                    new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
                for k in new_ingredient_params
            }

            model_str = f"model{i}"
            # Evaluate new model
            torch.save(potential_greedy_soup_params, os.path.join(args.results, "greedy_soup_temp.pt"))
            path_to_temp_results = f"{args.results}/greedy_soup_temp.pt"
            os.system(f"CUDA_VISIBLE_DEVICES=0 python3 ./latent-diffusion/main.py -r {path_to_temp_results} -l {args.results + '/' + model_str} -b {args.config}")

            # Read in new loss value
            result_metrics = pd.read_csv(get_csv(results_dir, model_str))
                #os.path.join(args.results, f"{model}testtube/version_0/metrics.csv"))
            held_out_loss = result_metrics['val/loss'][0]

            # 
            print(f'Potential greedy soup val acc {held_out_loss}, best so far {best_val_loss_so_far}.')
            if held_out_loss < best_val_loss_so_far:
                greedy_soup_ingredients.append(sorted_models[i])
                best_val_loss_so_far = held_out_loss
                greedy_soup_params = potential_greedy_soup_params
                print(f'Adding to soup. New soup is {greedy_soup_ingredients}')

        torch.save(greedy_soup_params, os.path.join(args.results, "greedy_soup_model.pt"))
        os.system(f"CUDA_VISIBLE_DEVICES=0 python3 ./latent-diffusion/main.py -r {args.results + 'greedy_soup_model.pt'} -l {args.results} -b {args.config}")
        
                
            

        
        

    
    
    

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
