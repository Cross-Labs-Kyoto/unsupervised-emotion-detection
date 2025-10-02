#!/usr/bin/env python3
import numpy as np
from scipy.io import loadmat
from pywt import wavedec
from matplotlib import pyplot as plt
from loguru import logger

MIN_NB_SAMPLES = 17152  # Gathered by analyzing the dataset. Corresponds to 67 seconds of video

mat = loadmat('DREAMER.mat', squeeze_me=True)  # squeeze_me=True required otherwise, you have to play with a lot of [0, 0]
dreamer = mat['DREAMER']
#print(dreamer.dtype)
nb_subs = dreamer['noOfSubjects'].item()
nb_vids = dreamer['noOfVideoSequences'].item()
sample_rate = dreamer['ECG_SamplingRate']

kernel = 5 * sample_rate
stride = 2 * sample_rate
nb_segments = int((MIN_NB_SAMPLES - kernel) / stride + 1)
nb_chans = 2

# See here for infor on structures in Numpy: https://numpy.org/doc/stable/reference/routines.rec.html
dreamer_data = dreamer['Data'].item()
# 'Age', 'Gender', 'EEG', 'ECG', 'ScoreValence', 'ScoreArousal', 'ScoreDominance' x 23 participants

data = np.zeros((nb_subs, nb_vids, nb_chans, MIN_NB_SAMPLES))
labels = np.zeros((nb_subs, nb_vids, 3))
for part_id, part in enumerate(dreamer_data):
    #print(part.dtype)
    score_vals = part['ScoreValence'].item()
    score_arou = part['ScoreArousal'].item()
    score_dom = part['ScoreDominance'].item()
    labels[part_id] = np.stack([score_vals, score_arou, score_dom], axis=-1)

    ecg = part['ECG'].item()
    baseline = ecg['baseline'].item()
    stimuli = ecg['stimuli'].item()
    data[part_id] = np.stack([s[0:MIN_NB_SAMPLES].transpose() for s in stimuli], axis=0)

# data shape: (sub, vid, chan, nb_points)
# label shape: (sub, vid, 3) where 3 is for Valence, arousal, dominance
print(labels.shape, data.shape)

features = np.zeros((nb_subs, nb_vids, nb_chans, nb_segments, 248))  # 248 is the size of the approximated coefficients at level 3
for sub_id in range(nb_subs):
    for vid_id in range(nb_vids):
        for chan_id in range(2):
            for idx in range(nb_segments):
                ca, *cds = wavedec(data[sub_id, vid_id, chan_id, idx*stride:(idx*stride)+kernel], 'coif17', level=3)
                features[sub_id, vid_id, chan_id, idx] = ca

features = features.reshape(nb_subs, -1, nb_chans * 248)
d_min = features.min(axis=1, keepdims=True)
d_max = features.max(axis=1, keepdims=True)
features = (features - d_min) / (d_max - d_min)
print(features.shape, features.min(), features.max())

nb_samples = np.ones(nb_vids) * nb_segments
