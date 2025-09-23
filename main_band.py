#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from loguru import logger

from datasets import load_data, EegDataset, TripletSampler, DatasetType
from networks import ContrastiveLSTM, ContrastiveFC
from settings import FACED, WIN_SIZE, STRIDE, BANDS


if __name__ == "__main__":
    # Declare command line interface
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=100,
                        help='The number of epochs to train/validate the network for.')
    parser.add_argument('-w', '--weights', dest='weights_file', type=Path, default=None,
                        help='The relative path to the pre-trained weights.')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=32,
                        help='The size of a batch to be fed to the network for training/inference.')
    parser.add_argument('-l', '--l_rate', dest='l_rate', type=float, default=1e-4,
                        help='The rate of weight adaptation.')
    parser.add_argument('-d', '--dropout', dest='dropout', type=float, default=0.25,
                        help='The ratio of neurons to deactivate in a layer during training.')
    parser.add_argument('-i', '--infer', dest='infer', action='store_true',
                        help='A flag indicating whether to extract features from the dataset, or train/test the network.')
    parser.add_argument('-o', '--output', dest='out_file', type=Path, default=None,
                        help='The relative path to the file in which to write the extracted feature vectors. This is only taken into account when `-i/--infer` is specified.')
    parser.add_argument('--psd', dest='psd', action='store_true',
                        help='A flag indicating whether to use the Power Spectral Density features, or the Differential Entropy.')
    parser.add_argument('--lstm', dest='lstm', action='store_true',
                        help='A flag indicating whether to use an LSTM-based architecture, instead of a Fully Connected one.')

    args = parser.parse_args()

    # Instantiate model
    if args.lstm:
        logger.info("Loading model - LSTM")
        model = ContrastiveLSTM(in_size=FACED['channels'] * len(BANDS), hidden_size=60,
                                out_size=30, l_rate=args.l_rate, batch_size=args.batch_size, dropout=args.dropout)
    else:
        logger.info("Loading model - FC")
        model = ContrastiveFC(in_size=120, out_size=30, hid_sizes=[100, 100, 50], l_rate=args.l_rate, batch_size=args.batch_size, dropout=args.dropout)
        WIN_SIZE = 1
        STRIDE = 1

    # Load model weight if necessary
    if args.weights_file is not None:
        weights_path = args.weights_file.expanduser().resolve()
        if weights_path.exists() and weights_path.is_file():
            logger.info(f'Loading weights from: {weights_path}')
            model.load_state_dict(torch.load(weights_path, weights_only=True))

    # Load data
    logger.info("Loading data")
    if args.psd:
        data, n_samples, n_segs, n_subs = load_data(WIN_SIZE, STRIDE, DatasetType.PSD)
    else:
        data, n_samples, n_segs, n_subs = load_data(WIN_SIZE, STRIDE, DatasetType.DE)


    if args.infer:
        # Build dataset
        logger.info("Creating dataset")
        data = data.reshape(-1, data.shape[-1])
        emo_ds = EegDataset(data, WIN_SIZE, STRIDE, n_subs, n_segs)
        emo_dl = DataLoader(emo_ds, shuffle=False, drop_last=False, batch_size=args.batch_size)

        # Do the thing
        logger.info("Inference - Here we go!")
        model.inference(emo_dl, args.out_file)
    else:
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
                              batch_sampler=TripletSampler(len(train_sub), args.batch_size, n_samples), num_workers=4)
        val_dl = DataLoader(val_ds,
                            batch_sampler=TripletSampler(len(val_sub), args.batch_size, n_samples), num_workers=4)

        # Train and validate
        model.train_net(train_dl, val_dl, args.epochs, patience=15)

        logger.info("Test - Here we go!")
        # Build dataset and associated loader
        logger.info("Creating datasets")
        test_ds = EegDataset(data[test_sub].reshape(-1, data.shape[-1]),
                             WIN_SIZE, STRIDE, len(test_sub), n_segs)
        test_dl = DataLoader(test_ds,
                             batch_sampler=TripletSampler(len(test_sub), args.batch_size, n_samples),
                             num_workers=4)

        # Test
        model.test_net(test_dl)
