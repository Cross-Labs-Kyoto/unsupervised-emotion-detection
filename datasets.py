#!/usr/bin/env python3
import pickle
from itertools import combinations
from collections import defaultdict
from enum import IntEnum, auto
import warnings
import numpy as np
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


def load_data(timeLen, timeStep, ds_type):
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
        logger.debug(f'Loading data from: {data_path}')
    elif ds_type == DatasetType.DE:
        data_path = DATA_DIR.joinpath('FACED', 'EEG_Features', 'DE')
        logger.debug(f'Loading data from: {data_path}')
    elif ds_type == DatasetType.PDS:
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
            data[idx] = data_sub[:,:-2]  # The last two channels are ignored

    # data shape :(sub, vid, chn, fs * sec) -> fs * sec = nb_points
    logger.debug(f'data loaded: {data.shape}')

    # Reshape the data
    n_subs = data.shape[0]
    if ds_type == DatasetType.CLISA:
        data = np.transpose(data, (0,1,3,2)).reshape(n_subs, -1, FACED['channels'])
    else:
        # Drop the first band since it is considered non-relevant for emotion detection
        data = data[:, :, :, :, 1:]
        data = np.transpose(data, (0,1,3,2,4)).reshape(n_subs, -1, FACED['channels'] * len(BANDS))

    logger.debug(f'data reshaped: {data.shape}')

    return data, n_samples, n_segs, n_subs


class ClisaDataset(Dataset):
    def __init__(self, data, timeLen, timeStep, n_segs):
        self.data = data.transpose() # nb_channels, tot_nb_points (nb_participants * nb_vids * nb_points)
        logger.debug(f'Dataset shape: {self.data.shape}')

        self.timeLen = timeLen
        self.timeStep = timeStep
        self.n_segs = n_segs
        self.fs = FACED['sample_freq']
        # Given that the kernel and stride do not perfectly divide the duration of a single time series, compute the duration that will be ignored at the end
        self.n_samples_remain_each = FACED['duration'] - n_segs * timeStep

    def __len__(self):
        return int((self.data.shape[-1] / (self.fs * FACED['duration'])) * self.n_segs)

    def __getitem__(self, idx):
        # Based on the given index, extract the right segment for all channels
        one_seq = self.data[:, int((idx * self.timeStep + self.n_samples_remain_each * np.floor(idx / self.n_segs)) * self.fs):int((idx * self.timeStep + self.timeLen + self.n_samples_remain_each * np.floor(idx / self.n_segs)) * self.fs)]

        return torch.FloatTensor(one_seq)

class EegDataset(Dataset):
    def __init__(self, data, timeLen, timeStep, n_segs):
        self.data = data.transpose() # nb_channels, tot_nb_points (nb_participants * nb_vids * nb_points)
        logger.debug(f'Dataset shape: {self.data.shape}')

        self.timeLen = timeLen
        self.timeStep = timeStep
        self.n_segs = n_segs
        # Given that the kernel and stride do not perfectly divide the duration of a single time series, compute the duration that will be ignored at the end
        self.n_samples_remain_each = FACED['duration'] - n_segs * timeStep

    def __len__(self):
        return int((self.data.shape[-1] / FACED['duration']) * self.n_segs)

    def __getitem__(self, idx):
        # Based on the given index, extract the right segment for all channels
        one_seq = self.data[:, int(idx * self.timeStep + self.n_samples_remain_each * np.floor(idx / self.n_segs)):int(idx * self.timeStep + self.timeLen + self.n_samples_remain_each * np.floor(idx / self.n_segs))]

        return torch.FloatTensor(one_seq)

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
        self.idx_to_labels = [0] * 3
        for i in range(1,4):
            self.idx_to_labels.extend([i] * 3)
        self.idx_to_labels.extend([4] * 4)
        for i in range(5,9):
            self.idx_to_labels.extend([i] * 3)

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
