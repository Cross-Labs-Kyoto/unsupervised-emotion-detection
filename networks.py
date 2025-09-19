#!/usr/bin/env python3
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from loguru import logger

from settings import DEVICE, WEIGHT_DIR


def stratified_norm(x, batch_size):
    """Compute the [stratified norm](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.626277/full) of the given input."""

    chunks = torch.split(x, batch_size, dim=0)
    out = x.clone()

    for i, chk in enumerate(chunks):
        m = torch.mean(chk, dim=(0, 1), keepdim=True)  # Compute mean across video for same participant, and same channel
        s = torch.std(chk, dim=(0, 1), keepdim=True)  # Compute std across video for same participant, and same channel
        out[i*batch_size:(i+1)*batch_size] = (chk - m) / (s + 1e-3)

    return out


def min_max_norm(x, batch_size):
    chunks = torch.split(x, batch_size, dim=0)
    out = x.clone()

    for i, chk in enumerate(chunks):
        vmin = torch.amin(chk, dim=(0, 1), keepdim=True)  # Compute min value across video for same participant, and same channel
        vmax = torch.amax(chk, dim=(0, 1), keepdim=True)  # Compute std across video for same participant, and same channel
        out[i*batch_size:(i+1)*batch_size] = (chk - vmin) / (vmax - vmin)

    return out


class Contrastive(nn.Module):
    """Define a lstm-based network that learns to generate informative representations through contrastive learning."""

    def __init__(self, in_size, hidden_size, out_size, nb_lstm=1,  l_rate=1e-4, batch_size=1, dropout=0, bidir=False, device=DEVICE):
        """Instantiate the lstm-based network, define the output, and declare the optimizer and loss."""

        super().__init__()
        # Define the lstm layers
        self._lstm = nn.LSTM(in_size, hidden_size, num_layers=nb_lstm, batch_first=True, dropout=dropout, bidirectional=bidir, device=device)

        # Declare a feature layer
        if bidir:
            in_feats = 2 * hidden_size
        else:
            in_feats = hidden_size

        self._fc1 = nn.Sequential(
            nn.Dropout(0.25, inplace=True),
            nn.Linear(in_feats, in_feats, device=device),
            nn.ReLU(inplace=True),
        )

        self._fc2 = nn.Sequential(
            nn.Dropout(0.25, inplace=True),
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

        # TODO: Multitaper features

        if not infer:
            # Head
            out = self._head(out.to(self._device))

        return out

    def train_net(self, ds, sampler, epochs, patience=20):
        # Keep track of the minimum validation loss and patience
        min_loss = None
        curr_patience = patience

        # Split the dataset into training and validation
        train_ds, val_ds = random_split(ds, [0.9, 0.1])
        # Instantiate the dataload with triplet sampler
        train_dl = DataLoader(train_ds, batch_sampler=sampler, num_workers=4)
        val_dl = DataLoader(val_ds, batch_sampler=sampler, num_workers=4)

        for epoch in epochs:
            # Train
            self.train()
            train_loss = 0
            for batch, _ in train_dl:
                batch = batch.permute((0, 2, 1)).to(self._device)  # N, L, chan

                preds = self(batch)
                anch, pos, neg = preds[:, :self._batch_size], preds[:, self._batch_size:2*self._batch_size], preds[:, 2*self._batch_size:]

                self._optim.zero_grad()
                loss = self._loss(anch, pos, neg)
                loss.backward()
                self._optim.step()

                train_loss += loss.item()

            train_loss /= len(train_dl)

            # And validate
            self = self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch, labels in val_dl:
                    batch = batch.permute((0, 2, 1)).to(self._device)  # N, L, chan
                    preds = self(batch)
                    anch, pos, neg = preds[:, :batch_size], preds[:, batch_size:2*batch_size], preds[:, 2*batch_size:]

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

    def test_net(self, ds, sampler):
        # Instantiate dataloader
        test_dl = DataLoader(ds, batch_sampler=sampler, num_workers=4)

        # Test
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for batch, _ in test_dl:
                batch = batch.permute((0, 2, 1)).to(self._device)

                preds = self(batch)
                anch, pos, neg = preds[:, :self._batch_size], preds[:, self._batch_size:2*self._batch_size], preds[:, 2*self._batch_size:]
                test_loss += self._loss(anch, pos, neg)

        # Print final stats
        logger.info(f'Test loss: {test_loss / len(test_dl)}')

    def inference(self, ds, batch_size=64):
        # TODO: Use regular sampler
        # TODO: Store features in h5py file? Or pyarrow? Or pytable?
        pass
