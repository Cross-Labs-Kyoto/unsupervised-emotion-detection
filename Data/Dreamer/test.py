#!/usr/bin/env python3
import numpy as np
from scipy.io import loadmat
from pywt import wavedec
from matplotlib import pyplot as plt

MIN_NB_SAMPLES = 17152  # Gathered by analyzing the dataset. Corresponds to 67 seconds of video

mat = loadmat('DREAMER.mat', squeeze_me=True)  # squeeze_me=True required otherwise, you have to play with a lot of [0, 0]
dreamer = mat['DREAMER']
#print(dreamer.dtype)
nb_subs = dreamer['noOfSubjects'].item()
nb_vids = dreamer['noOfVideoSequences'].item()
sample_rate = dreamer['ECG_SamplingRate']


# See here for infor on structures in Numpy: https://numpy.org/doc/stable/reference/routines.rec.html
dreamer_data = dreamer['Data'].item()
# 'Age', 'Gender', 'EEG', 'ECG', 'ScoreValence', 'ScoreArousal', 'ScoreDominance' x 23 participants

data = np.zeros((nb_subs, nb_vids, 2, MIN_NB_SAMPLES))
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
    
print(labels.shape, data.shape)
ca, *cds = wavedec(data[0, 0, 0], 'coif17', level=5)
plt.plot(ca, color='b')
plt.plot(data[0, 0, 0], color='r')
plt.savefig('test.png')
# data shape: (sub, vid, chan, nb_points)
# label shape: (sub, vid, 3) where 3 is for Valence, arousal, dominance
