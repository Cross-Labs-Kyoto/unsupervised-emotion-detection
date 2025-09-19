#!/usr/bin/env python3
from pathlib import Path
from torch import cuda, device


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.joinpath('Data')
WEIGHT_DIR = ROOT_DIR.joinpath('Weights')
DEVICE = device('cuda') if cuda.is_available() else device('cpu')

FACED = {'channels': 30,  # Two channels are ignored
         'duration': 30,
         'nb_vids': 28,
         'sample_freq': 250}
FACED['nb_points'] = FACED['sample_freq'] * FACED['duration']

BANDS = ([4, 7], [8, 13], [14, 30], [31, 50])
