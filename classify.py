#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from h5py import File
from torch.utils.data import DataLoader, random_split
from loguru import logger
from datasets import ClassificationDataset
from network import ClassifierFC


BATCH_SIZE = 64

if __name__ == "__main__":
    # Define command line interface
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', dest='db_file', type=Path, required=True,
                        help='The relative path to the file containing the feature vectors.')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=100,
                        help='The number of epochs to train/validate the network for.')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=32,
                        help='The size of a batch to be fed to the network for training/inference.')
    parser.add_argument('-l', '--l_rate', dest='l_rate', type=float, default=1e-4,
                        help='The rate of weight adaptation.')

    args = parser.parse_args()
    
    # TODO: Build the classifier
    logger.info('Creating model')
    classifier = ClassifierFC(in_size=, out_size=9, hid_sizes=[30, 30], l_rate=args.l_rate)
    logger.debug(classifier)

    # Load the dataset from file
    logger.info('Create dataset')
    with File(args.db_file, 'r') as db_file:
        vects = db_file['vectors'][:]
        labels = db_file['labels'][:]

    # Build the dataset and dataloader
    ds = ClassificationDataset(vects, labels)
    train, test_ds = random_split(ds, lengths=[0.8, 0.2])
    train_ds, val_ds = random_split(train, lengths=[0.9, 0.1])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)

    # Train
    classifier.train_net(train_dl, val_dl, args.epochs, patience=15)

    # Test
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=4)
    classifier.test_net(test_dl)
