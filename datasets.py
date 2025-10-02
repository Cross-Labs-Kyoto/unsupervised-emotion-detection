#!/usr/bin/env python3
import pickle
from itertools import combinations
from collections import defaultdict
from enum import IntEnum, auto
import warnings
import numpy as np
from scipy.io import loadmat
from pywt import wavedec
import torch
from torch.utils.data import Dataset, Sampler

from settings import DATA_DIR, FACED, BANDS
from loguru import logger

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class DatasetType(IntEnum):
    CLISA = auto()
    DE = auto()
    PSD = auto()


def get_labels(n_subs=None, n_segs=None):
    labels = [0] * 3
    for i in range(1,9):
        labels.extend([i] * 3)

    labels = np.array(labels, dtype=int)

    if n_segs is not None:
        label_segs = []
        for i in range(len(labels)):
            label_segs.extend([labels[i]] * n_segs)

        labels = np.array(label_segs, dtype=int)

    if n_subs is not None:
        labels = np.tile(labels, n_subs)

    return labels


def load_data_eeg(timeLen, timeStep, ds_type):
    """
    Loads the FACED - Clisa data from files, builds the associated list of
    labels, and compute various meta-data about the dataset.

    Parameters
    ----------
    timeLen: int
        The length of the kernel in second.

    timeStep: int
        The length of the stride in second.

    ds_type: IntEnum
        The dataset to load in memory.

    """

    # Compute the number of segments available in each time series
    n_segs = int((FACED['duration'] - timeLen) / timeStep + 1)  # Same formula as output dim of convolution layer
    # Compute the number of segments across all time series
    n_samples = np.ones(FACED['nb_vids']) * n_segs

    # Get the time series from file
    if ds_type == DatasetType.CLISA:
        data_path = DATA_DIR.joinpath('FACED', 'Clisa_data')
    elif ds_type == DatasetType.DE:
        data_path = DATA_DIR.joinpath('FACED', 'EEG_Features', 'DE')
    elif ds_type == DatasetType.PSD:
        data_path = DATA_DIR.joinpath('FACED', 'EEG_Features', 'PSD')
    logger.debug(f'Loading data from: {data_path}')
    data_paths = [itm for itm in sorted(data_path.iterdir()) if itm.exists() and not itm.is_dir()]

    if ds_type == DatasetType.CLISA:
        data = np.zeros((len(data_paths), FACED['nb_vids'], FACED['channels'],
                         FACED['nb_points']))
    else:
        data = np.zeros((len(data_paths), FACED['nb_vids'], FACED['channels'], FACED['duration'],
                         len(BANDS) + 1))
    for idx, path in enumerate(data_paths):
        with path.open('rb') as f:
            data_sub = pickle.load(f)
            data[idx] = np.delete(data_sub[:, :-2], 12, axis=0)  # Keep the dataset balanced by removing the additional 4 category, the last two channels are ignored since they are not EEG

    # data shape :(sub, vid, chn, fs * sec) -> fs * sec = nb_points
    logger.debug(f'data loaded: {data.shape}')

    # Reshape the data
    n_subs = data.shape[0]
    if ds_type == DatasetType.CLISA:
        data = np.transpose(data, (0,1,3,2))

        # Min_max normalization
        d_min = data.min(axis=1, keepdims=True)
        d_max = data.max(axis=1, keepdims=True)
        data = ((data - d_min) / (d_max - d_min)).reshape(n_subs, -1, FACED['channels'])

    else:
        # Drop the first band since it is considered non-relevant for emotion detection
        data = data[:, :, :, :, 1:]
        data = np.transpose(data, (0,1,3,2,4)).reshape(n_subs, -1, FACED['channels'] * len(BANDS))

        # Min_max normalization
        d_min = data.min(axis=1, keepdims=True)
        d_max = data.max(axis=1, keepdims=True)
        data = (data - d_min) / (d_max - d_min)
    logger.debug(f'data reshaped: {data.shape}')

    return data, n_samples, n_segs, n_subs


def load_data_ecg(timeLen, timeStep): 
    min_nb_samples = 17152  # Gathered by analyzing the dataset. Corresponds to 67 seconds of video

    mat = loadmat(DATA_DIR.joinpath('Dreamer', 'DREAMER.mat'), squeeze_me=True)  # squeeze_me=True required otherwise, you have to play with a lot of [0, 0]
    dreamer = mat['DREAMER']
    #print(dreamer.dtype)
    nb_subs = dreamer['noOfSubjects'].item()
    nb_vids = dreamer['noOfVideoSequences'].item()
    sample_rate = dreamer['ECG_SamplingRate']

    kernel = timeLen * sample_rate
    stride = timeStep * sample_rate
    nb_segments = int((min_nb_samples - kernel) / stride + 1)
    nb_chans = 2

    # See here for infor on structures in Numpy: https://numpy.org/doc/stable/reference/routines.rec.html
    dreamer_data = dreamer['Data'].item()
    # 'Age', 'Gender', 'EEG', 'ECG', 'ScoreValence', 'ScoreArousal', 'ScoreDominance' x 23 participants

    data = np.zeros((nb_subs, nb_vids, nb_chans, min_nb_samples))
    emo_labels = np.zeros((nb_subs, nb_vids, 3))
    for part_id, part in enumerate(dreamer_data):
        #print(part.dtype)
        score_vals = part['ScoreValence'].item()
        score_arou = part['ScoreArousal'].item()
        score_dom = part['ScoreDominance'].item()
        emo_labels[part_id] = np.stack([score_vals, score_arou, score_dom], axis=-1)

        ecg = part['ECG'].item()
        baseline = ecg['baseline'].item()
        stimuli = ecg['stimuli'].item()
        data[part_id] = np.stack([s[0:min_nb_samples].transpose() for s in stimuli], axis=0)

    emo_labels = emo_labels.reshape(-1, 3) 
    # data shape: (sub, vid, chan, nb_points)
    # label shape: (sub, vid, 3) where 3 is for Valence, arousal, dominance

    features = np.zeros((nb_subs, nb_vids, nb_chans, nb_segments, 248))  # 248 is the size of the approximated coefficients at level 3
    vid_labels = np.zeros((nb_subs, nb_vids, nb_segments))  # The video labels and emotion labels do not map to each other 1:1. We are assuming that the video elicit the emotion it's supposed to.
    for sub_id in range(nb_subs):
        for vid_id in range(nb_vids):
            vid_labels[sub_id, vid_id] = [vid_id] * nb_segments
            for chan_id in range(2):
                for idx in range(nb_segments):
                    ca, *cds = wavedec(data[sub_id, vid_id, chan_id, idx*stride:(idx*stride)+kernel], 'coif17', level=3)
                    features[sub_id, vid_id, chan_id, idx] = ca

    vid_labels = vid_labels.reshape(-1, nb_segments)

    # feature shape: nb_subs, nb_vids * nb_segments, nb_chans * 248
    features = features.transpose((0, 1, 3, 2, 4)).reshape(nb_subs, -1, nb_chans * 248)
    d_min = features.min(axis=1, keepdims=True)
    d_max = features.max(axis=1, keepdims=True)
    features = (features - d_min) / (d_max - d_min)
    features = features.reshape(-1, nb_chans * 248)


    nb_samples = np.ones(nb_vids) * nb_segments

    return features, vid_labels.flatten(), nb_samples, nb_segments, nb_subs


class ClisaDataset(Dataset):
    def __init__(self, data, timeLen, timeStep, n_subs, n_segs):
        self.data = data.transpose() # nb_channels, tot_nb_points (nb_participants * nb_vids * nb_points)
        logger.debug(f'Dataset shape: {self.data.shape}')

        self.timeLen = timeLen
        self.timeStep = timeStep
        self.n_segs = n_segs
        self.fs = FACED['sample_freq']
        # Given that the kernel and stride do not perfectly divide the duration of a single time series, compute the duration that will be ignored at the end
        self.n_samples_remain_each = FACED['duration'] - n_segs * timeStep

        self.labels = get_labels(n_subs, n_segs)

    def __len__(self):
        return int((self.data.shape[-1] / (self.fs * FACED['duration'])) * self.n_segs)

    def __getitem__(self, idx):
        # Based on the given index, extract the right segment for all channels
        one_seq = self.data[:, int((idx * self.timeStep + self.n_samples_remain_each * np.floor(idx / self.n_segs)) * self.fs):int((idx * self.timeStep + self.timeLen + self.n_samples_remain_each * np.floor(idx / self.n_segs)) * self.fs)]

        return torch.FloatTensor(one_seq), self.labels[idx]


class EegDataset(Dataset):
    def __init__(self, data, timeLen, timeStep, n_subs, n_segs):
        self.data = data.transpose() # nb_channels, tot_nb_points (nb_participants * nb_vids * nb_points)
        logger.debug(f'Dataset shape: {self.data.shape}')

        self.timeLen = timeLen
        self.timeStep = timeStep
        self.n_segs = n_segs
        # Given that the kernel and stride do not perfectly divide the duration of a single time series, compute the duration that will be ignored at the end
        self.n_samples_remain_each = FACED['duration'] - n_segs * timeStep

        self.labels = get_labels(n_subs, n_segs)

    def __len__(self):
        return int((self.data.shape[-1] / FACED['duration']) * self.n_segs)

    def __getitem__(self, idx):
        # Based on the given index, extract the right segment for all channels
        one_seq = self.data[:, int(idx * self.timeStep + self.n_samples_remain_each * np.floor(idx / self.n_segs)):int(idx * self.timeStep + self.timeLen + self.n_samples_remain_each * np.floor(idx / self.n_segs))]

        return torch.FloatTensor(one_seq), self.labels[idx]


class TripletSampler(Sampler[list[int]]):
    def __init__(self, nb_subs, batch_size, nb_samples):
        self.batch_size = batch_size
        self.nb_samples_cum = np.concatenate((np.array([0]), np.cumsum(nb_samples)))
        assert self.batch_size >= len(self.nb_samples_cum) - 1, f"The batch size ({batch_size}) should be greater than the number of videos ({len(self.nb_samples_cum) - 1})."


        self.nb_samples_cum_set = set(range(len(self.nb_samples_cum) - 2))
        self.nb_samples_per_trial = int(batch_size / len(nb_samples))
        self.n_per = int(np.sum(nb_samples))  # samples per sub = nb_samples * nb_vids

        self.subs_set = set(range(nb_subs))
        self.sub_pairs = list(combinations(range(nb_subs), 2))

        # Build the mapping video idx -> label
        self.idx_to_labels = get_labels().tolist()

        # Build the reverse mapping label -> video idxs
        self.labels_to_idx = defaultdict(list)
        for idx, label in enumerate(self.idx_to_labels):
            self.labels_to_idx[label].append(idx)

    def __len__(self):
        return len(self.sub_pairs)

    def __iter__(self):
        for sub_a, sub_p in self.sub_pairs:
            # Initialize the list of indices for the anchor/positive and negative segments
            ind_ap = np.zeros(0)
            ind_n = np.zeros(0)

            # Choose a negative participant different from the anchor and positive participants
            sub_n = np.random.choice(list(self.subs_set - set([sub_a, sub_p])))

            for i in range(len(self.nb_samples_cum) - 2):
                # Get segment indices for the anchor/positive
                ind_one = np.random.choice(np.arange(self.nb_samples_cum[i], self.nb_samples_cum[i + 1]),
                                           self.nb_samples_per_trial, replace=False)
                ind_ap = np.concatenate((ind_ap, ind_one))

                # Gather all the indices that belong to the same label as the current one
                a_is = self.labels_to_idx[self.idx_to_labels[i]]
                # Choose a negative index different from i
                n_i = np.random.choice(list(self.nb_samples_cum_set - set(a_is)))
                # Get segment indices for negative
                ind_two = np.random.choice(np.arange(self.nb_samples_cum[n_i], self.nb_samples_cum[n_i + 1]),
                                           self.nb_samples_per_trial, replace=False)
                ind_n = np.concatenate((ind_n, ind_two))

            i = len(self.nb_samples_cum) - 2
            ind_one = np.random.choice(np.arange(self.nb_samples_cum[i], self.nb_samples_cum[i+1]),
                                       int(self.batch_size - len(ind_ap)),
                                       replace=False)
            ind_ap = np.concatenate((ind_ap, ind_one))

            # Choose a negative index different from i
            a_is = self.labels_to_idx[self.idx_to_labels[i]]
            n_i = np.random.choice(list(self.nb_samples_cum_set - set(a_is)))
            # Get segment indices for negative
            ind_two = np.random.choice(np.arange(self.nb_samples_cum[n_i], self.nb_samples_cum[n_i + 1]),
                                       int(self.batch_size - len(ind_n)),
                                       replace=False)
            ind_n = np.concatenate((ind_n, ind_two))

            # Offset the segment indices for each participant
            ind_a = ind_ap + self.n_per * sub_a
            ind_p = ind_ap + self.n_per * sub_p
            ind_n = ind_n + self.n_per * sub_n

            # Concatenate all segment indices into one list
            batch = torch.LongTensor(np.concatenate((ind_a, ind_p, ind_n)))

            yield batch


class ClassificationDataset(Dataset):
    def __init__(self, data, labels):
        self._data = data
        self._labels = labels

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        return self._data[idx], self._labels[idx]
