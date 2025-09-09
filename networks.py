#!/usr/bin/env python3
import torch
from torch import nn
from torch import cuda
from torch.optim import AdamW
from torch.utils.data import Dataloader, random_split
from loguru import logger


DEVICE = torch.device('cuda') if cuda.is_available() else torch.device('cpu')


class LSTM(nn.Module):
    """Declare a common LSTM-based module for dealing with EEG time series."""

    def __init__(self, in_size, hidden_size, out_size, nb_lstm=1, dropout=0, bidir=False, device=DEVICE):
        """Initialize an Lstm-based network with a partial head."""

        super().__init__()
        # Define the lstm layers
        self._lstm = nn.LSTM(in_size, hidden_size, num_layers=nb_lstm, batch_first=True, dropout=dropout, bidirectional=bidir, device=device)

        # TODO: What about normalization?
        # Declare a feature layer
        if bidir:
            in_feats = 2 * hidden_size
        else:
            in_feats = hidden_size
        self._feat = nn.Sequential(
            nn.Linear(in_feats, 64, device=device),
            nn.Dropout(0.25, inplace=True),
            nn.ReLU(inplace=True),
        )

        # TODO: What about normalization?
        # Declare the fully connected part of the head
        # The output activation is left out on purpose to allow for customization down the line
        self._fc = nn.Sequential(
            nn.Linear(64, 64, device=device),
            nn.Dropout(0.25, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_size, device=device)
        )

    def forward(self, x):
        """Propagate the given input through the lstm and partial head."""

        out, _ = self._lstm(x)
        return self._fc(self._feat(out[:, -1, :]))  # Only feed the last hidden state to the head


class Classifier(LSTM):
    """Define an Lstm-based network for classifying EEG time series into emotion classes."""

    def __init__(self, in_size, hidden_size, out_size, nb_lstm=1, l_rate=1e-3, dropout=0, bidir=False, device=DEVICE):
        """Instantiate the Lstm structure and declare a task specific output layer."""

        # Create the basic structure
        super().__init__(in_size, hidden_size, out_size, nb_lstm=1, dropout=0, bidir=False, device=DEVICE)

        # Declare the output layer
        self._out = nn.Softmax(dim=1)

        # Declare the loss and optimizer
        self._loss = nn.CrossEntropyLoss()
        self._optim = AdamW(self.parameters(), lr=l_rate, amsgrad=True)

    def forward(self, x):
        """Propagate the given input through the lstm-based common body, and apply softmax to the output."""
        x = super()(x)
        return self._out(x)

    def train_net(self, ds, epochs, batch_size=64, stop_crit=0.01, patience=20):
        # Keep track of the minimum loss and current patience
        min_loss = None
        curr_patience = patience

        # Split the dataset into train and validation dataloader
        train_ds, val_ds = random_split(ds, [0.9, 0.1])
        train_dl = Dataloader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
        val_dl = Dataloader(val_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)

        for epoch in epochs:
            # Ensure the network is in training mode
            self = self.train()
            # Train
            train_loss = 0
            for batch, labels in train_dl:
                self._optim.zero_grad()
                preds = self(batch)
                loss = self._loss(preds, labels)
                loss.backward()
                self._optim.step()

                train_loss += loss.item()

            train_loss /= len(train_dl)

            # Ensure the network is in evaluation mode
            self = self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch, labels in val_dl:
                    preds = self(batch)
                    val_loss += self._loss(preds, labels).item()

            # Check stop criterion
            val_loss /= len(val_dl)
            if min_loss is None or val_loss < min_loss:
                min_loss = val_loss
                curr_patience = patience
            else:
                curr_patience -= 1

            if curr_patience <= 0:
                break

            # Display some statistics
            logger.info(f'{epoch},{train_loss},{val_loss}')

    def test_net(self, ds, batch_size=64):
        # Instantiate the dataloader
        test_dl = Dataloader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1)

        # Ensure the network is in evaluation mode
        self = self.eval()
        test_loss = 0
        with torch.no_grad():
            for batch, labels in test_dl:
                preds = self(batch)
                test_loss += self._loss(preds, labels).item()

        # Display some statistics
        logger.info(f'Test loss: {test_loss / len(test_dl)}')

# TODO: Define classifier based on generic class
# TODO: Define contrastive learning NN based on generic class

# TODO: Both classified and contrastive learning NN should declare their own Train, Test, and Infer methods => Include loss, optimizer, and path to weight file
