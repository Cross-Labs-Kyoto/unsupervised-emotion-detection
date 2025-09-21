#!/usr/bin/env python3
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from mne.time_frequency import psd_array_multitaper
from scipy.integrate import simpson
from h5py import File
from loguru import logger

from settings import DEVICE, WEIGHT_DIR, ROOT_DIR, BANDS, FACED


def stratified_norm(x, batch_size):
    """Compute the [stratified norm](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.626277/full) of the given input."""

    out = x.clone()
    chunks = torch.split(x, batch_size, dim=0)

    for i, chk in enumerate(chunks):
        if i == 0:
            m = torch.mean(chk, dim=(0, 1), keepdim=True)  # Compute mean across video for same participant, and same channel
            s = torch.std(chk, dim=(0, 1), keepdim=True)  # Compute std across video for same participant, and same channel
        else:
            m = torch.mean(chk, dim=1, keepdim=True)  # Compute mean across video for same participant, and same channel
            s = torch.std(chk, dim=1, keepdim=True)  # Compute std across video for same participant, and same channel
        out[i*batch_size:(i+1)*batch_size] = (chk - m) / (s + 1e-3)

    return out


def min_max_norm(x, batch_size):
    out = x.clone()
    chunks = torch.split(x, batch_size, dim=0)

    for i, chk in enumerate(chunks):
        if i == 0:
            vmin = torch.amin(chk, dim=(0, 1), keepdim=True)  # Compute min value across video for same participant, and same channel
            vmax = torch.amax(chk, dim=(0, 1), keepdim=True)  # Compute max across video for same participant, and same channel
        else:
            vmin = torch.amin(chk, dim=1, keepdim=True)  # Compute min value across video for same participant, and same channel
            vmax = torch.amax(chk, dim=1, keepdim=True)  # Compute max across video for same participant, and same channel
        out[i*batch_size:(i+1)*batch_size] = (chk - vmin) / (vmax - vmin)

    return out


def multitaper(x, batch_size):
    out = torch.zeros((x.shape[0], x.shape[-1] * len(BANDS))) # batch, seq, feat
    x = x.permute((0, 2, 1)).cpu().numpy()  # batch, feat, seq
    for vid in range(x.shape[0]):
        for feat in range(x.shape[1]):
            data = x[vid, feat]
            for i, band in enumerate(BANDS):
                low, high = band
                psd_trial, freqs = psd_array_multitaper(data, FACED['sample_freq'],
                                                        adaptive=True,
                                                        n_jobs=-1,
                                                        normalization='full',
                                                        verbose=False)
                freq_res = freqs[1] - freqs[0]
                idx_band = np.logical_and(freqs >= low, freqs <= high)

                out[vid, i + feat] = simpson(psd_trial[idx_band], dx=freq_res)  # Band power

    return out

class ContrastiveLSTM(nn.Module):
    """Define a lstm-based network that learns to generate informative representations through contrastive learning."""

    def __init__(self, in_size, hidden_size, out_size, nb_lstm=1,  l_rate=1e-4, batch_size=1, dropout=0, bidir=False, device=DEVICE):
        """Instantiate the lstm-based network, define the output, and declare the optimizer and loss."""

        super().__init__()
        # Define the lstm layers
        self._lstm = nn.LSTM(in_size, hidden_size, num_layers=nb_lstm, batch_first=True, dropout=dropout if nb_lstm > 1 else 0, bidirectional=bidir, device=device)

        # Declare a feature layer
        if bidir:
            in_feats = 2 * hidden_size
        else:
            in_feats = hidden_size

        self._fc1 = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(in_feats, in_feats, device=device),
            nn.ReLU(inplace=True),
        )

        self._fc2 = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(in_feats, in_feats, device=device),
            nn.ReLU(inplace=True),
        )

        # Declare the fully connected part of the head
        # The output activation is left out on purpose to allow for customization down the line
        self._head = nn.Linear(in_feats, out_size, device=device)

        # Declare loss and optimizer
        self._loss = nn.TripletMarginLoss()
        self._optim = AdamW(self.parameters(), lr=l_rate, amsgrad=True)

        self._device = device
        self._batch_size = batch_size

    def forward(self, x, infer=False):
        """Propagate the given input through the network, and apply stratified normalization to the output."""

        # Min-Max normalization
        x = min_max_norm(x, self._batch_size)
        # Lstm
        out, _ = self._lstm(x)
        out = stratified_norm(out, self._batch_size)

        # Fc1
        out = self._fc1(out)
        out = stratified_norm(out, self._batch_size)

        # Fc2
        out = self._fc2(out)
        out = stratified_norm(out, self._batch_size)

        if infer:
            # Extract Multitaper features
            out = multitaper(out, self._batch_size)
        else:
            # Head
            out = self._head(out[:, -1, :])

        return out

    def train_net(self, train_dl, val_dl, epochs, patience=20):
        # Keep track of the minimum validation loss and patience
        min_loss = None
        curr_patience = patience

        for epoch in range(epochs):
            # Train
            self.train()
            train_loss = 0
            for batch in train_dl:
                batch = batch.permute((0, 2, 1)).to(self._device)  # N, L, chan

                preds = self(batch)
                anch, pos, neg = preds[:self._batch_size], preds[self._batch_size:2*self._batch_size], preds[2*self._batch_size:]

                self._optim.zero_grad()
                loss = self._loss(anch, pos, neg)
                loss.backward()
                self._optim.step()

                train_loss += loss.item()

            train_loss /= len(train_dl)

            # And validate
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dl:
                    batch = batch.permute((0, 2, 1)).to(self._device)  # N, L, chan
                    preds = self(batch)
                    anch, pos, neg = preds[:self._batch_size], preds[self._batch_size:2*self._batch_size], preds[2*self._batch_size:]

                    val_loss += self._loss(anch, pos, neg).item()

            # Check stop criterion
            val_loss /= len(val_dl)

            if min_loss is None or val_loss < min_loss:
                min_loss = val_loss
                curr_patience = patience

                # Save weights to file
                torch.save(self.state_dict(), WEIGHT_DIR.joinpath('contrastive_lstm.pth'))
            else:
                curr_patience -= 1

            if curr_patience <= 0:
                break

            # Display some statistics
            logger.info(f'{epoch},{train_loss},{val_loss}')

    def test_net(self, test_dl):
        # Load the best weights
        if WEIGHT_DIR.joinpath('contrastive_lstm.pth').exists():
            self.load_state_dict(torch.load(WEIGHT_DIR.joinpath('contrastive_lstm.pth')), weights_only=True)

        # Test
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_dl:
                batch = batch.permute((0, 2, 1)).to(self._device)

                preds = self(batch)
                anch, pos, neg = preds[:self._batch_size], preds[self._batch_size:2*self._batch_size], preds[2*self._batch_size:]
                test_loss += self._loss(anch, pos, neg)

        # Print final stats
        logger.info(f'Test loss: {test_loss / len(test_dl)}')

    def inference(self, dl, out_file):
        if out_file is None:
            out_file = ROOT_DIR.joinpath('feature_vectors.h5')

        # Create a new hdf5 dataset to store the feature vectors
        h5_ds = None
        with File(out_file, 'w') as h5_file:
            # Extract the feature vectors from the emotion dataset
            self.eval()
            with torch.no_grad():
                for batch in dl:
                    batch = batch.permute((0, 2, 1)).to(self._device)
                    vects = self(batch, infer=True).cpu().numpy()
                    # Append the feature vectors to the dataset
                    if h5_ds is None:
                        h5_ds = h5_file.create_dataset('default',
                                                       data=vects,
                                                       shape=vects.shape,
                                                       maxshape=(None, *vects.shape[1:]),
                                                       chunks=True)
                    else:
                        h5_ds.resize(h5_ds.shape[0] + vects.shape[0], axis=0)
                        h5_ds[-vects.shape[0]:] = vects
