#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:16:45 2022

@author: alexey.osipov
"""

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils import data
from transforms import ToTensor
import GeoDataset
from deeplabv3plus import deeplabv3_resnet50, deeplabv3_mobilenet
import cv2


def validate(model, loader, device):
    with torch.no_grad():
        val_iter = iter(loader)
        numerator = 0
        denominator = 0
        for i in range(len(loader)):
            print(i, numerator, denominator)
            val_info = next(val_iter)
            images = val_info['features']
            labels = val_info['masks']
            files = val_info['filenames']
            images = images.to(device, dtype=torch.float32)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            for j in range(len(images)):
                pred = preds[j]
                pred = cv2.resize(pred.astype('float32'), dsize = (32, 32))
                pred = pred.reshape(-1, 1)
                target = targets[j]
                target = target.reshape(-1, 1)
                numerator += sum([int((pred[k] > 0) and (target[k] > 0)) for k in range(32*32)])
                denominator += sum([int((pred[k] > 0) or (target[k] > 0)) for k in range(32*32)])
                pred = pd.DataFrame(pred, columns=['prediction'])
                target = pd.DataFrame(target, columns=['target'])
                filename = files[j]
                pred.to_csv('/home/alexey.osipov/playground/eco_geo/prototype/data/results/'
                            + 'pred_' + filename)
                target.to_csv('/home/alexey.osipov/playground/eco_geo/prototype/data/results/'
                            + 'target_' + filename)
    return numerator, denominator


def main(): 
    target_col = 'F'
    features_standartization_info = {'min': GeoDataset.MIN_VALS_FOR_F,
                                     'max': GeoDataset.MAX_VALS_FOR_F,
                                     'mean': GeoDataset.MEAN_VALS_FOR_F,
                                     'sd': GeoDataset.SD_VALS_FOR_F}
    data_transforms = {'test': ToTensor()}
    root_dir = '/home/alexey.osipov/playground/eco_geo/prototype/'
    geoDataset_test = GeoDataset.GeoDataset('test.txt',
                                            target_col,
                                            GeoDataset.FEATURE_COLS_FOR_F,
                                            features_standartization_info,
                                            root_dir,
                                            transform=data_transforms['test'])
    geoDataset_test.return_raw_masks = True
    geo_datasets = {'test': geoDataset_test}
    dataloaders = {'test': data.DataLoader(geo_datasets['test'], batch_size=16,
                                           shuffle=True, num_workers=0,
                                           drop_last=True)}
    print("Dataset: %s, Test set: %d" %
         ('geodataset', len(dataloaders['test'])))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = '/home/alexey.osipov/playground/eco_geo/prototype/model_for_F_2.pth'
    
    random_seed = 239
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    output_stride = 8
    model = deeplabv3_resnet50(num_classes = 2, output_stride = output_stride)
        
    checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    print("validation...")
    model.eval()
    numerator, denominator = validate(model=model,
                                      loader=dataloaders['test'],
                                      device=device)
    print('numerator')
    print(numerator)
    print('denominator')
    print(denominator)
