#!/usr/bin/env python3
import pickle
from itertools import combinations
from collections import defaultdict
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from settings import DATA_DIR, FACED
from loguru import logger

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_labels():
    # Build the list of labels for a single time series
    label = [0] * 3
    for i in range(1,4):
        label.extend([i] * 3)
    label.extend([4] * 4)
    for i in range(5,9):
        label.extend([i] * 3)

    return label


def load_data(timeLen, timeStep):
    """
    Loads the FACED - Clisa data from files, builds the associated list of
    labels, and compute various meta-data about the dataset.

    Parameters
    ----------
    timeLen: int
        The length of the kernel in second.

    timeStep: int
        The length of the stride in second.

    """

    # Compute the number of segments available in each time series
    n_segs = int((FACED['duration'] - timeLen) / timeStep + 1)  # Same formula as output dim of convolution layer
    # Compute the number of segments across all time series
    n_samples = np.ones(FACED['nb_vids']) * n_segs

    logger.debug(f'{n_segs} segments/time series, {n_samples.sum().astype(int)} total segments')

    # Get the time series from file
    data_path = DATA_DIR.joinpath('FACED', 'Clisa_data')
    logger.debug(f'Loading data from: {data_path}')
    data_paths = [itm for itm in sorted(data_path.iterdir()) if itm.exists() and not itm.is_dir()]

    data = np.zeros((len(data_paths), FACED['nb_vids'], FACED['channels'],
                     FACED['nb_points']))
    for idx, path in enumerate(data_paths):
        with path.open('rb') as f:
            data_sub = pickle.load(f)
            data[idx,:,:,:] = data_sub[:,:-2,:]  # The last two channels are ignored

    # data shape :(sub, vid, chn, fs * sec) -> fs * sec = nb_points
    logger.debug(f'data loaded: {data.shape}')

    # Reshape the data
    n_subs = data.shape[0]
    data = np.transpose(data, (0,1,3,2)).reshape(n_subs, -1, FACED['channels'])
    logger.debug(f'data reshaped: {data.shape}')

    label = get_labels()

    # Extend the list of labels to all time series
    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]] * n_segs

    return data, label_repeat, n_samples, n_segs, n_subs


class EmotionDataset(Dataset):
    def __init__(self, data, label, timeLen, timeStep, n_segs):
        self.data = data.transpose() # nb_channels, tot_nb_points (nb_participants * nb_vids * nb_points)
        logger.debug(f'Emotion dataset shape: {self.data.shape}')

        self.timeLen = timeLen
        self.timeStep = timeStep
        self.n_segs = n_segs
        self.fs = FACED['sample_freq']
        self.label = torch.from_numpy(label)
        # Given that the kernel and stride do not perfectly divide the duration of a single time series, compute the duration that will be ignored at the end
        self.n_samples_remain_each = FACED['duration'] - n_segs * timeStep

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Based on the given index, extract the right segment for all channels
        one_seq = self.data[:, int((idx * self.timeStep + self.n_samples_remain_each * np.floor(idx / self.n_segs)) * self.fs):int((idx * self.timeStep + self.timeLen + self.n_samples_remain_each * np.floor(idx / self.n_segs)) * self.fs)]

        # Get the corresponding label
        one_label = self.label[idx]

        return torch.FloatTensor(one_seq), one_label


class TripletSampler(Sampler[list[int]]):
    def __init__(self, nb_subs, batch_size, nb_samples, labels):
        self.batch_size = batch_size
        self.nb_samples_cum = np.concatenate((np.array([0]), np.cumsum(nb_samples)))
        assert self.batch_size >= len(self.nb_samples_cum) - 1, f"The batch size ({batch_size}) should be greater than the number of videos ({len(self.nb_samples_cum) - 1})."


        self.nb_samples_cum_set = set(range(len(self.nb_samples_cum) - 2))
        self.nb_samples_per_trial = int(batch_size / len(nb_samples))
        self.n_per = int(np.sum(nb_samples))

        self.subs_set = set(range(nb_subs))
        self.sub_pairs = combinations(range(nb_subs), 2)
        self.nb_sub_pairs = len(list(combinations(range(nb_subs), 2)))

        self.idx_to_labels = labels
        self.labels_to_idx = defaultdict(list)
        for idx, label in enumerate(labels):
            self.labels_to_idx[label].append(idx)

    def __len__(self):
        return self.nb_sub_pairs

    def __iter__(self):
        for sub_a, sub_p in self.sub_pairs:
            # Initialize the list of indices for the anchor/positive and negative segments
            ind_ap = np.zeros(0)
            ind_n = np.zeros(0)

            # Choose a negative participant different from the anchor and positive participants
            sub_n = np.random.choice(list(self.subs_set - set([sub_a, sub_p])))
            logger.debug(f'Participants: a - {sub_a}, p - {sub_p}, n - {sub_n}')

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

if __name__ == "__main__":
    kernel = 5
    stride = 2
    batch_size = 32

    data, label_repeat, n_samples, n_segs, n_subs = load_data(kernel, stride)
    logger.debug(n_samples)

    labels = np.tile(label_repeat, n_subs)
    data = data.reshape(-1, data.shape[-1])
    emo_ds = EmotionDataset(data, labels, kernel, stride, n_segs)

    ts = TripletSampler(n_subs, batch_size, n_samples, get_labels())

    dl = DataLoader(emo_ds, batch_sampler=ts)
    for seq, labels in dl:
        logger.debug(f'Input shape: {seq.shape}, Label shape: {labels.shape}')
        exit()
