import argparse
import os
import wget
import torch
import clip
import os
import json
import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as np

import torchvision.datasets as datasets

from datasets import ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA
from utils import get_model_from_sd, test_model_on_dataset


# Evaluate individual models ------------------------------------------
def individualModels(base_model, model_paths, resultsFile):
    if os.path.exists(resultsFile):
        os.remove(resultsFile)
        
    for j, model_path in enumerate(model_paths):
        assert os.path.exists(model_path)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model = get_model_from_sd(state_dict, base_model)

        results = {'model_name' : f'model_{j}'}
        # Note: ImageNet2p is the held-out minival set from ImageNet train that we use.
        # It is called 2p for 2 percent of ImageNet, or 26k images.
        # See utils on how this dataset is handled slightly differently.
        for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:

            print(f'Evaluating model {j} of {NUM_MODELS - 1} on {dataset_cls.__name__}.')

            dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
            accuracy = test_model_on_dataset(model, dataset)
            results[dataset_cls.__name__] = accuracy
            print(accuracy)

        with open(resultsFile, 'a+') as f:
            f.write(json.dumps(results) + '\n')

            
# Uniform Soup Function -----------------------------------------------
def uniformSoup(base_model, model_paths, numModels, resultsFile):
    # If result file already exists, remove it
    if os.path.exists(resultsFile):
        os.remove(resultsFile)

    # Create the uniform soup sequentially to not overload memory
    for j, model_path in enumerate(model_paths):
        print(f'Adding model {j} of {numModels - 1} to uniform soup.')

        assert os.path.exists(model_path)

        # Load model in from file
        state_dict = torch.load(model_path)
        if j == 0:
            uniform_soup = {k : v * (1./numModels) for k, v in state_dict.items()}
        else:
            uniform_soup = {k : v * (1./numModels) + uniform_soup[k] for k, v in state_dict.items()}

    # Create the model from uniform soup state_dict
    model = get_model_from_sd(uniform_soup, base_model)

    # Test each model and save the results to a JSON file
    results = {'model_name' : f'uniform_soup'}
    for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:
        print(f'Evaluating on {dataset_cls.__name__}.')
        
        dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
        accuracy = test_model_on_dataset(model, dataset)
        results[dataset_cls.__name__] = accuracy
        print(accuracy)

    # Output file
    with open(resultsFile, 'a+') as f:
        f.write(json.dumps(results) + '\n')

    return uniform_soup

# Greedy Soup function ------------------------------------------------
def greedySoup(base_model, indyResultsFile, resultFile, numModels):
    if os.path.exists(resultFile):
        os.remove(resultFile)
            
    # Sort models by decreasing accuracy on the held-out validation set ImageNet2p
    # (We call the held out-val set ImageNet2p because it is 2 percent of ImageNet train)
    individual_model_db = pd.read_json(indyResultsFile, lines=True)
    individual_model_val_accs = {}
        
    for _, row in individual_model_db.iterrows():
        individual_model_val_accs[row['model_name']] = row['ImageNet2p']
            
    individual_model_val_accs = sorted(individual_model_val_accs.items(), key=operator.itemgetter(1))
    individual_model_val_accs.reverse()
    sorted_models = [x[0] for x in individual_model_val_accs]
    
    # Start the soup by using the first ingredient.
    greedy_soup_ingredients = [sorted_models[0]]
    greedy_soup_params = torch.load(os.path.join(args.model_location, f'{sorted_models[0]}.pt'))
    best_val_acc_so_far = individual_model_val_accs[0][1]
    held_out_val_set = ImageNet2p(preprocess, args.data_location, args.batch_size, args.workers)

        # Now, iterate through all models and consider adding them to the greedy soup.
    for i in range(1, numModels):
        print(f'Testing model {i} of {numModels}')

        # Get the potential greedy soup, which consists of the greedy soup with the new model added.
        new_ingredient_params = torch.load(os.path.join(args.model_location, f'{sorted_models[i]}.pt'))
        num_ingredients = len(greedy_soup_ingredients)
        potential_greedy_soup_params = {
            k : greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) + 
                new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
            for k in new_ingredient_params
        }

        # Run the potential greedy soup on the held-out val set.
        model = get_model_from_sd(potential_greedy_soup_params, base_model)
        held_out_val_accuracy = test_model_on_dataset(model, held_out_val_set)

        # If accuracy on the held-out val set increases, add the new model to the greedy soup.
        print(f'Potential greedy soup val acc {held_out_val_accuracy}, best so far {best_val_acc_so_far}.')
        if held_out_val_accuracy > best_val_acc_so_far:
            greedy_soup_ingredients.append(sorted_models[i])
            best_val_acc_so_far = held_out_val_accuracy
            greedy_soup_params = potential_greedy_soup_params
            print(f'Adding to soup. New soup is {greedy_soup_ingredients}')

    # Finally, evaluate the greedy soup.
    model = get_model_from_sd(greedy_soup_params, base_model)
    results = {'model_name' : f'greedy_soup'}
    for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:
        print(f'Evaluating on {dataset_cls.__name__}.')
        dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
        accuracy = test_model_on_dataset(model, dataset)
        results[dataset_cls.__name__] = accuracy
        print(accuracy)

    with open(resultFile, 'a+') as f:
        f.write(json.dumps(results) + '\n')
