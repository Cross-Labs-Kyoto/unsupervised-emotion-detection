#!/usr/bin/env python3
from pathlib import Path
from torch import cuda, device


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.joinpath('Data')
DEVICE = device('cuda') if cuda.is_available() else device('cpu')

FACED = {'channels': 30,  # Two channels are ignored
         'duration': 30,
         'nb_vids': 28,
         'sample_freq': 250}
FACED['nb_points'] = FACED['sample_freq'] * FACED['duration']
