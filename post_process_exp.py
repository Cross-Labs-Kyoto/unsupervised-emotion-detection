#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from h5py import File
import umap
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from loguru import logger

from datasets import load_data, DatasetType, ClassificationDataset, EegDataset
from networks import ContrastiveLSTM, ContrastiveFC, ClassifierFC
from settings import FACED, WIN_SIZE, STRIDE, BANDS, WEIGHT_DIR, ROOT_DIR


if __name__ == "__main__":
    # Constant
    batch_size = 32
    epochs=100
    dropout=0

    # Parmeters to iterate over
    hid_lstms = [5, 10, 20, 40, 60]
    hid_fcs = [[20, 20], [30, 30], [40, 40], [50, 50], [100, 100]]
    out_sizes = [2, 3, 5, 10]
    l_rates = [1e-3, 5e-4, 1e-4]
    psds = [False, True]

    for psd in psds:
        for l_rate in l_rates:
            for out_size in out_sizes:
                for hid_fc in hid_fcs:
                    for hid_lstm in hid_lstms:
                        ####################
                        # Feature extraction
                        ####################

                        # Instantiate model
                        logger.info(f"Loading model - psd: {psd} l_rate: {l_rate} out_size: {out_size} hid_fc: {hid_fc} hid_lstm: {hid_lstm}")
                        weight_name = f'contrastive_lstm_{psd}_{l_rate}_{out_size}_{hid_fc}_{hid_lstm}.pth'
                        weight_path = WEIGHT_DIR.joinpath(weight_name)

                        if not weight_path.exists() or not weight_path.is_file():  # Just ignore non-existant weight files
                            logger.info(f"Weight file does not exist, moving on: {weight_path}")
                            continue
                        # TODO: out_size should not be hardcoded (out_size=out_size)
                        model = ContrastiveLSTM(in_size=FACED['channels'] * len(BANDS), hid_lstm=hid_lstm, hid_fc=hid_fc,
                                                out_size=3, l_rate=l_rate, batch_size=batch_size, dropout=dropout, weight_file=WEIGHT_DIR.joinpath(weight_name))

                        # Load the weights
                        model.load_state_dict(torch.load(weight_path, weights_only=True))
                        model.eval()

                        # Load data
                        if psd:
                            data, n_samples, n_segs, n_subs = load_data(WIN_SIZE, STRIDE, DatasetType.PSD)
                        else:
                            data, n_samples, n_segs, n_subs = load_data(WIN_SIZE, STRIDE, DatasetType.DE)

                        # Create the dataset and dataloader
                        data = data.reshape(-1, data.shape[-1])
                        emo_ds = EegDataset(data, WIN_SIZE, STRIDE, n_subs, n_segs)
                        emo_dl = DataLoader(emo_ds, shuffle=False, drop_last=False, batch_size=batch_size)

                        # Extract emotion related information from the data
                        feat_file = ROOT_DIR.joinpath('Results', f'features_{psd}_{l_rate}_{out_size}_{hid_fc}_{hid_lstm}.h5')
                        model.inference(emo_dl, feat_file)

                        ####################
                        # Classify
                        ####################

                        # Instantiate the classifier
                        # TODO: in_size should not be hardcoded (in_size=out_size)
                        classifier = ClassifierFC(in_size=3, out_size=9, hid_sizes=[30, 30], l_rate=1e-3)

                        # Create classification database
                        with File(feat_file, 'r') as db_file:
                            vects = db_file['vectors'][:]
                            labels = db_file['labels'][:].reshape(-1, 1)

                        # One-hot encode the labels
                        encoder = OneHotEncoder(dtype=labels.dtype)
                        onehot_labels = encoder.fit_transform(labels).toarray().astype(np.float64)

                        # Build the dataset and dataloader
                        ds = ClassificationDataset(vects, onehot_labels)
                        train, test_ds = random_split(ds, lengths=[0.8, 0.2])
                        train_ds, val_ds = random_split(train, lengths=[0.9, 0.1])

                        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=4)
                        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

                        # Train
                        classifier.train_net(train_dl, val_dl, 500, patience=50)

                        # Test
                        test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=4)
                        classifier.test_net(test_dl)

                        with File(feat_file, 'a') as db_file:
                            ####################
                            # Cluster
                            ####################

                            # Import feature vectors
                            feat_vecs = db_file['vectors']

                            # Import the ground truth labels
                            gt_labels = db_file['labels']

                            # Cluster
                            cluster = MiniBatchKMeans(n_clusters=9)  # n_clusters=9 because FACED datatset has 9 emotion categories
                            #cluster = HDBSCAN(min_cluster_size=10, min_samples=30, n_jobs=8)

                            # Get the cluster labels
                            clst_labels = cluster.fit_predict(feat_vecs)

                            # Store the cluster labels
                            db_file.create_dataset('clusters', data=clst_labels, shape=clst_labels.shape,
                                                   dtype=clst_labels.dtype, chunks=True)

                            # Compute clustering performance
                            mis = adjusted_mutual_info_score(gt_labels, clst_labels)
                            rs = adjusted_rand_score(gt_labels, clst_labels)

                            logger.info(f'K-Means scores: {mis}, {rs}')

                            ####################
                            # plot
                            ####################
    
                            if feat_vecs.shape[-1] > 3:
                                reducer = umap.UMAP()
                                # Fit and transform the feature vectors
                                fit_ds = reducer.fit_transform(feat_vecs)
                            else:
                                fit_ds = feat_vecs

                            if fit_ds.shape[-1] == 2:
                                plt.scatter(fit_ds[:, 0], fit_ds[:, 1], c=clst_labels, cmap='tab10')
                                plt.savefig(ROOT_DIR.joinpath('Results', f'features_{psd}_{l_rate}_{out_size}_{hid_fc}_{hid_lstm}_clusters.png'))

                                plt.scatter(fit_ds[:, 0], fit_ds[:, 1], c=gt_labels, cmap='tab10')
                                plt.savefig(ROOT_DIR.joinpath('Results', f'features_{psd}_{l_rate}_{out_size}_{hid_fc}_{hid_lstm}_ground_truth.png'))
                            else:
                                plt.scatter(fit_ds[:, 0], fit_ds[:, 1], fit_ds[:, 2], c=clst_labels, cmap='tab10')
                                plt.savefig(ROOT_DIR.joinpath('Results', f'features_{psd}_{l_rate}_{out_size}_{hid_fc}_{hid_lstm}_clusters.png'))

                                plt.scatter(fit_ds[:, 0], fit_ds[:, 1], fit_ds[:, 2], c=gt_labels, cmap='tab10')
                                plt.savefig(ROOT_DIR.joinpath('Results', f'features_{psd}_{l_rate}_{out_size}_{hid_fc}_{hid_lstm}_ground_truth.png'))

