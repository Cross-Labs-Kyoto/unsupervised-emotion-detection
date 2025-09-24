#!/usr/bin/env python3
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from h5py import File
from loguru import logger
from settings import ROOT_DIR

if __name__ == "__main__":
    # TODO: Declare command line interface to specify input/output files, and type of clustering to use (k-means or hdbscan)
    with File(ROOT_DIR.joinpath('Results', 'test_de.h5'), 'a') as db_file:
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
