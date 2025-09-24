#!/usr/bin/env python3
from pathlib import Path
from torch import cuda, device


# Declare paths to common folders
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.joinpath('Data')
WEIGHT_DIR = ROOT_DIR.joinpath('Weights')
DEVICE = device('cuda') if cuda.is_available() else device('cpu')

# Faced dataset metadata
FACED = {'channels': 30,  # Two channels are ignored
         'duration': 30,
         'nb_vids': 27,  # One sub-set of category 4 removed to balance the dataset
         'sample_freq': 250}
FACED['nb_points'] = FACED['sample_freq'] * FACED['duration']
BANDS = ([4, 8], [8, 14], [14, 31], [31, 50])

# Configure training/inference process
WIN_SIZE = 5
STRIDE = 2
