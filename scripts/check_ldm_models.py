#! /usr/bin/env python3

import torch
import json
from pathlib import Path

if __name__ == '__main__':
    models_dir = Path('latent-diffusion/models')
    ckpt_paths = [Path(c) for c in models_dir.glob('**/*.ckpt')]

    # Iterate over LDM models and compare to other models.
    # The goal is to group together models that have the same structure, as these models
    # can be easily used with model soups.
    soup_groups = []
    while len(ckpt_paths) > 1:
        ckpt_path = ckpt_paths.pop()
        print(ckpt_path)
        ckpt_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']

        # Compare to all models remaining in the list
        soup_group = []
        for compared_path in ckpt_paths:
            compared_dict = torch.load(compared_path, map_location='cpu')['state_dict']

            # Check if structure is the same
            soupable = True
            for k in ckpt_dict.keys():
                soupable &= k in compared_dict and ckpt_dict[k].size() == compared_dict[k].size()
                if not soupable:
                    break
            if soupable:
                soup_group.append(compared_path)

        if len(soup_group):
            [ckpt_paths.remove(soup_ckpt) for soup_ckpt in soup_group]
            soup_group.insert(0, ckpt_path)
            soup_groups.append(soup_group)
        print(soup_group)
