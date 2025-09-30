#!/usr/bin/env python3
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from loguru import logger

from datasets import load_data, EegDataset, TripletSampler, DatasetType
from networks import ContrastiveLSTM, ContrastiveFC
from settings import FACED, WIN_SIZE, STRIDE, BANDS, WEIGHT_DIR


if __name__ == "__main__":
    # Constant
    batch_size = 32
    epochs=100
    dropout=0

    # Parmeters to iterate over
    hid_lstms = [5, 10, 20, 40, 60]
    hid_fcs = [[20, 20], [30, 30], [40, 40], [50, 50], [100, 100]]
    out_sizes = [2, 3, 5, 10]
    l_rates = [3e-4, 1e-4]
    psds = [False, True]

    for psd in psds:
        for l_rate in l_rates:
            for out_size in out_sizes:
                for hid_fc in hid_fcs:
                    for hid_lstm in hid_lstms:
                        # Instantiate model
                        logger.info(f"Loading model - psd: {psd} l_rate: {l_rate} out_size: {out_size} hid_fc: {hid_fc} hid_lstm: {hid_lstm}")
                        weight_name = f'contrastive_lstm_{psd}_{l_rate}_{out_size}_{";".join(map(str, hid_fc))}_{hid_lstm}.pth'
                        model = ContrastiveLSTM(in_size=FACED['channels'] * len(BANDS), hid_lstm=hid_lstm, hid_fc=hid_fc,
                                                out_size=3, l_rate=l_rate, batch_size=batch_size, dropout=dropout, weight_file=WEIGHT_DIR.joinpath(weight_name))

                        # Load data
                        logger.info("Loading data")
                        if psd:
                            data, n_samples, n_segs, n_subs = load_data(WIN_SIZE, STRIDE, DatasetType.PSD)
                        else:
                            data, n_samples, n_segs, n_subs = load_data(WIN_SIZE, STRIDE, DatasetType.DE)


                        # Split the subs between train and test sets
                        subs_idx = set(range(n_subs))

                        test_sub = np.random.choice(list(subs_idx), int(len(subs_idx) * 0.1)).tolist()

                        subs_idx.difference_update(set(test_sub))
                        val_sub = np.random.choice(list(subs_idx), int(len(subs_idx) * 0.1)).tolist()

                        train_sub = list(subs_idx.difference(set(val_sub)))

                        logger.info("Train/Validate - Here we go!")
                        # Build datasets and associated loaders
                        logger.info("Creating datasets")
                        train_ds = EegDataset(data[train_sub].reshape(-1, data.shape[-1]),
                                              WIN_SIZE, STRIDE, len(train_sub), n_segs)
                        val_ds = EegDataset(data[val_sub].reshape(-1, data.shape[-1]),
                                            WIN_SIZE, STRIDE, len(val_sub), n_segs)

                        # Instantiate the corresponding loaders
                        train_dl = DataLoader(train_ds,
                                              batch_sampler=TripletSampler(len(train_sub), batch_size, n_samples), num_workers=4)
                        val_dl = DataLoader(val_ds,
                                            batch_sampler=TripletSampler(len(val_sub), batch_size, n_samples), num_workers=4)

                        # Train and validate
                        model.train_net(train_dl, val_dl, epochs, patience=15)

                        logger.info("Test - Here we go!")
                        # Build dataset and associated loader
                        logger.info("Creating datasets")
                        test_ds = EegDataset(data[test_sub].reshape(-1, data.shape[-1]),
                                             WIN_SIZE, STRIDE, len(test_sub), n_segs)
                        test_dl = DataLoader(test_ds,
                                             batch_sampler=TripletSampler(len(test_sub), batch_size, n_samples),
                                             num_workers=4)

                        # Test
                        model.test_net(test_dl)
