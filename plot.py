#!/usr/bin/env python3
from pathlib import Path
from argparse import ArgumentParser
from h5py import File
import umap
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from loguru import logger

if __name__ == "__main__":
    # Define command line interface
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', dest='db_file', type=Path, required=True,
                        help='The relative path to the file containing the feature vectors.')
    parser.add_argument('-o', '--output', dest='plt_file', type=Path, required=True,
                        help='The relative path in which to save the plot.')
    parser.add_argument('--tsne', dest='tsne', action='store_true', help='A flag indicating to use T-Sne for dimensionality reduction, instead of UMAP.')

    args = parser.parse_args()

    # Make sure path to feature vectors is valid
    db_file = args.db_file.expanduser().resolve()
    assert db_file.exists() and db_file.is_file(), f"The provided path, either does not exist or is not a file: {db_file}."

    # Instantiate UMAP object
    if args.tsne:
        reducer = TSNE(metric='cosine')
    else:
        reducer = umap.UMAP(metric='cosine')
    
    with File(db_file, 'r') as h5_f:
        logger.info('Loading feature vectors')
        # Import feature vectors
        h5_ds = h5_f['default']

        if args.tsne:
            logger.info('Dimensionality reduction - T-SNE')
        else:
            logger.info('Dimensionality reduction - UMAP')

        # Fit and transform the feature vectors
        fit_ds = reducer.fit_transform(h5_ds)

        logger.info('Plotting results')
        plt.scatter(fit_ds[:, 0], fit_ds[:, 1])
        plt.savefig(args.plt_file.expanduser().resolve())
