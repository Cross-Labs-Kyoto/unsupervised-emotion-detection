#!/usr/bin/env python3
from scipy.io import loadmat

mat = loadmat('DREAMER.mat', squeeze_me=True)  # squeeze_me=True required otherwise, you have to play with a lot of [0, 0]
dreamer = mat['DREAMER']
# See here for infor on structures in Numpy: https://numpy.org/doc/stable/reference/routines.rec.html
print(dreamer.dtype)
dreamer_data = dreamer['Data'].item()
# 'Age', 'Gender', 'EEG', 'ECG', 'ScoreValence', 'ScoreArousal', 'ScoreDominance' x 23 participants
print(dreamer_data.shape)
for part in dreamer_data:
    print(part['Gender'])
