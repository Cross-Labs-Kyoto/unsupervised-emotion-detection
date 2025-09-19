#!/usr/bin/env python3
from argparse import ArgumentParser
import numpy as np
from torch.utils.data import DataLoader
from loguru import logger

from datasets import load_data, get_labels, EmotionDataset, TripletSampler
from networks import Contrastive
from settings import FACED, DEVICE

# TODO: Define command line interface to switch between Train, Test, and Inference

# TODO: load_data
# TODO: create EmotionDataset
# TODO: create TripletSampler

# TODO: create network
# TODO: Load weights from file whenever necessary.

# TODO: Train/test/inference as requested

if __name__ == "__main__":
    kernel = 5
    stride = 2
    batch_size = 32

    # Instantiate model
    logger.info("Loading model")
    model = Contrastive(in_size=FACED['channels'], hidden_size=FACED['channels'] // 2, out_size=10, l_rate=0.0007, batch_size=batch_size, dropout=0.25)

    # Load data
    logger.info("Loading data")
    data, label_repeat, n_samples, n_segs, n_subs = load_data(kernel, stride)
    logger.debug(n_samples)

    # Build dataset
    logger.info("Creating dataset")
    labels = np.tile(label_repeat, n_subs)
    data = data.reshape(-1, data.shape[-1])
    emo_ds = EmotionDataset(data, labels, kernel, stride, n_segs)

    # Instantiate dataloader
    ts = TripletSampler(n_subs, batch_size, n_samples, get_labels())
    dl = DataLoader(emo_ds, batch_sampler=ts)
    logger.debug(f'Length dataloader: {len(dl)}')

    # Do the testing
    logger.info("Here we go.")
    for seq in dl:
        seq = seq.permute((0, 2, 1)).to(DEVICE)
        logger.debug(f'Input shape: {seq.shape}')

        pred = model(seq)
        logger.debug(f'Output shape: {pred.shape}')

        exit()
