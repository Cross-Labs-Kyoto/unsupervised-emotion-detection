#!/usr/bin/env python3
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, accuracy_score
from h5py import File
from loguru import logger

from settings import DEVICE, WEIGHT_DIR, ROOT_DIR, WIN_SIZE, FACED


def stratified_norm(x, batch_size):
    """Compute the [stratified norm](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.626277/full) of the given input."""

    out = x.clone()
    chunks = torch.split(x, batch_size, dim=0)

    if len(x.shape) > 2:
        for i, chk in enumerate(chunks):
            m = torch.mean(chk, dim=(0, 1), keepdim=True)  # Compute mean across video for same participant, and same channel
            s = torch.std(chk, dim=(0, 1), keepdim=True)  # Compute std across video for same participant, and same channel
            out[i*batch_size:(i+1)*batch_size] = (chk - m) / (s + 1e-8)
    else:
        for i, chk in enumerate(chunks):
            m = torch.mean(chk, dim=0, keepdim=True)  # Compute mean across video for same participant, and same channel
            s = torch.std(chk, dim=0, keepdim=True)  # Compute std across video for same participant, and same channel
            out[i*batch_size:(i+1)*batch_size] = (chk - m) / (s + 1e-8)

    return out


class ContrastiveLSTM(nn.Module):
    """Define a lstm-based network that learns to generate informative representations through contrastive learning."""

    def __init__(self, in_size, hid_lstm, hid_fc, out_size, nb_lstm=1,  l_rate=1e-4, batch_size=1, dropout=0.25, device=DEVICE, weight_file=WEIGHT_DIR.joinpath('contrastive_lstm.pth')):
        """Instantiate the lstm-based network, define the output, and declare the optimizer and loss."""

        super().__init__()
        # Define the lstm layers
        self._lstm = nn.LSTM(in_size, hid_lstm, num_layers=nb_lstm, batch_first=True, dropout=dropout if nb_lstm > 1 else 0, device=device)

        # Declare a feature layer
        #in_feats = hidden_size * WIN_SIZE * FACED['sample_freq']
        in_feats = hid_lstm

        in_dim = [in_feats] + hid_fc
        self._lays = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_f, out_f, device=device),
                nn.ReLU()
                ) for in_f, out_f in zip(in_dim, hid_fc)])

        # Declare the fully connected part of the head
        # The output activation is left out on purpose to allow for customization down the line
        self._head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid_fc[-1], out_size, device=device)
        )

        # Declare loss and optimizer
        self._loss = nn.TripletMarginLoss()
        self._optim = AdamW(self.parameters(), lr=l_rate, amsgrad=True)

        self._device = device
        self._batch_size = batch_size
        self._weight_file = weight_file

    def forward(self, x, infer=False):
        """Propagate the given input through the network, and apply stratified normalization to the output."""

        # Lstm
        out, _ = self._lstm(x)
        out = stratified_norm(out, self._batch_size)

        # Propagate through the common dense layers
        out = out[:, -1]
        for lay in self._lays:
            out = lay(out)
            out = stratified_norm(out, self._batch_size)

        #if not infer:
            # Head
        out = self._head(out)

        return out

    def train_net(self, train_dl, val_dl, epochs, patience=20):
        # Keep track of the minimum validation loss and patience
        sched_lr = CosineAnnealingLR(self._optim, epochs, eta_min=1e-7)
        min_loss = None
        curr_patience = patience

        for epoch in range(epochs):
            # Train
            self.train()
            train_loss = 0
            for batch, _ in train_dl:
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
                for batch, _ in val_dl:
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
                torch.save(self.state_dict(), self._weight_file)
            else:
                curr_patience -= 1

            if curr_patience <= 0:
                break

            # Display some statistics
            logger.info(f'{epoch},{train_loss},{val_loss}')
            sched_lr.step()

    def test_net(self, test_dl):
        # Load the best weights
        if self._weight_file.exists():
            self.load_state_dict(torch.load(self._weight_file, weights_only=True))

        # Test
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for batch, _ in test_dl:
                batch = batch.permute((0, 2, 1)).to(self._device)

                preds = self(batch)
                anch, pos, neg = preds[:self._batch_size], preds[self._batch_size:2*self._batch_size], preds[2*self._batch_size:]
                test_loss += self._loss(anch, pos, neg)

        # Print final stats
        logger.info(f'Test loss: {test_loss / len(test_dl)}')

    def inference(self, dl, out_file):
        if out_file is None:
            out_file = ROOT_DIR.joinpath('Results', 'feature_vectors.h5')
        else:
            out_file = ROOT_DIR.joinpath('Results', out_file)

        # Create a new hdf5 dataset to store the feature vectors
        h5_ds = None
        label_ds = None
        with File(out_file, 'w') as h5_file:
            # Extract the feature vectors from the emotion dataset
            self.eval()
            with torch.no_grad():
                for batch, labels in dl:
                    batch = batch.permute((0, 2, 1)).to(self._device)
                    vects = self(batch, infer=True).cpu().numpy()
                    # TODO: Store the labels alongside the feature vectors
                    # Append the feature vectors to the dataset
                    if h5_ds is None:
                        h5_ds = h5_file.create_dataset('vectors',
                                                       data=vects,
                                                       shape=vects.shape,
                                                       maxshape=(None, *vects.shape[1:]),
                                                       chunks=True)

                        label_ds = h5_file.create_dataset('labels',
                                                          data=labels,
                                                          shape=labels.shape,
                                                          maxshape=(None,),
                                                          chunks=True)
                    else:
                        h5_ds.resize(h5_ds.shape[0] + vects.shape[0], axis=0)
                        h5_ds[-vects.shape[0]:] = vects

                        label_ds.resize(label_ds.shape[0] + labels.shape[0], axis=0)
                        label_ds[-labels.shape[0]:] = labels


class ContrastiveFC(nn.Module):
    def __init__(self, in_size, out_size, hid_sizes, l_rate=1e-4, batch_size=1, dropout=0.25):
        super().__init__()

        # Build the body of the model
        in_feats = [in_size] + hid_sizes

        self._lays = nn.ModuleList([nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, out_f, device=DEVICE),
            nn.LeakyReLU()
        ) for in_f, out_f in zip(in_feats, hid_sizes)])

        # Build the head separately
        self._head = nn.Sequential(nn.Dropout(dropout),
                                   nn.Linear(hid_sizes[-1], out_size, device=DEVICE))

        # Declare the optimizer and loss
        self._optim = AdamW(self.parameters(), lr=l_rate, amsgrad=True)
        self._loss = nn.TripletMarginLoss()

        # Keep track of the batch size
        self._batch_size = batch_size

    def forward(self, x, infer=False):
        # Propagate the input through the model's body
        out = stratified_norm(x, self._batch_size)
        for lay in self._lays:
            out = lay(out)
            out = stratified_norm(out, self._batch_size)

        #if not infer:
            # Propagate through the head
        out = self._head(out)

        return out

    def train_net(self, train_dl, val_dl, epochs, patience=20):
        sched_lr = CosineAnnealingLR(self._optim, epochs, eta_min=1e-7)
        # Keep track of the minimum validation loss and patience
        min_loss = None
        curr_patience = patience

        for epoch in range(epochs):
            # Train
            self.train()
            train_loss = 0
            for batch, _ in train_dl:
                batch = batch.squeeze(2).to(DEVICE)  # N, FEAT

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
                for batch, _ in val_dl:
                    batch = batch.squeeze(2).to(DEVICE)  # N, L, chan
                    preds = self(batch)
                    anch, pos, neg = preds[:self._batch_size], preds[self._batch_size:2*self._batch_size], preds[2*self._batch_size:]

                    val_loss += self._loss(anch, pos, neg).item()

            # Check stop criterion
            val_loss /= len(val_dl)

            if min_loss is None or val_loss < min_loss:
                min_loss = val_loss
                curr_patience = patience

                # Save weights to file
                torch.save(self.state_dict(), WEIGHT_DIR.joinpath('contrastive_fc.pth'))
            else:
                curr_patience -= 1

            if curr_patience <= 0:
                break

            # Display some statistics
            logger.info(f'{epoch},{train_loss},{val_loss}')
            sched_lr.step()

    def test_net(self, test_dl):
        # Load the best weights
        if WEIGHT_DIR.joinpath('contrastive_fc.pth').exists():
            self.load_state_dict(torch.load(WEIGHT_DIR.joinpath('contrastive_fc.pth'), weights_only=True))

        # Test
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for batch, _ in test_dl:
                batch = batch.squeeze(2).to(DEVICE)  # N, FEAT

                preds = self(batch)
                anch, pos, neg = preds[:self._batch_size], preds[self._batch_size:2*self._batch_size], preds[2*self._batch_size:]
                test_loss += self._loss(anch, pos, neg)

        # Print final stats
        logger.info(f'Test loss: {test_loss / len(test_dl)}')

    def inference(self, dl, out_file):
        if out_file is None:
            out_file = ROOT_DIR.joinpath('Results', 'feature_vectors.h5')
        else:
            out_file = ROOT_DIR.joinpath('Results', out_file)

        # Create a new hdf5 dataset to store the feature vectors
        h5_ds = None
        label_ds = None
        with File(out_file, 'w') as h5_file:
            # Extract the feature vectors from the emotion dataset
            self.eval()
            with torch.no_grad():
                for batch, labels in dl:
                    batch = batch.squeeze(2).to(DEVICE)  # N, FEAT
                    vects = self(batch, infer=True).cpu().numpy()
                    # TODO: Store the labels alongside the feature vectors
                    # Append the feature vectors to the dataset
                    if h5_ds is None:
                        h5_ds = h5_file.create_dataset('vectors',
                                                       data=vects,
                                                       shape=vects.shape,
                                                       maxshape=(None, *vects.shape[1:]),
                                                       chunks=True)

                        label_ds = h5_file.create_dataset('labels',
                                                          data=labels,
                                                          shape=labels.shape,
                                                          maxshape=(None,),
                                                          chunks=True)
                    else:
                        h5_ds.resize(h5_ds.shape[0] + vects.shape[0], axis=0)
                        h5_ds[-vects.shape[0]:] = vects

                        label_ds.resize(label_ds.shape[0] + labels.shape[0], axis=0)
                        label_ds[-labels.shape[0]:] = labels


class ClassifierFC(nn.Module):
    def __init__(self, in_size, out_size, hid_sizes, l_rate=1e-4):
        super().__init__()
        in_feats = [in_size] + hid_sizes

        self._lays = nn.ModuleList([nn.Sequential(nn.Linear(in_f, out_f, device=DEVICE),
                                                  nn.ReLU(),
                                                  nn.LayerNorm(out_f, device=DEVICE)
                                                  ) for in_f, out_f in zip(in_feats, hid_sizes)])

        self._head = nn.Sequential(nn.Linear(hid_sizes[-1], out_size, device=DEVICE), nn.Softmax(dim=1))

        self._loss = nn.CrossEntropyLoss()
        self._optim = AdamW(self.parameters(), lr=l_rate, amsgrad=True)

    def forward(self, x):
        out = x
        for lay in self._lays:
            out = lay(out)
        return self._head(out)

    def train_net(self, train_dl, val_dl, epochs, patience=2):
        min_loss = None
        curr_patience = patience

        for epoch in range(epochs):
            # Train
            self.train()
            train_loss = 0
            for batch, labels in train_dl:
                batch = batch.to(DEVICE)
                labels = labels.to(DEVICE)

                preds = self(batch)

                self._optim.zero_grad()
                loss = self._loss(preds, labels)
                loss.backward()
                self._optim.step()

                train_loss += loss.item()

            train_loss /= len(train_dl)

            # And validate
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch, labels in val_dl:
                    batch = batch.to(DEVICE)
                    labels = labels.to(DEVICE)
                    preds = self(batch)

                    val_loss += self._loss(preds, labels).item()

            # Check stop criterion
            val_loss /= len(val_dl)

            if min_loss is None or val_loss < min_loss:
                min_loss = val_loss
                curr_patience = patience

                # Save weights to file
                torch.save(self.state_dict(), WEIGHT_DIR.joinpath('classifier_fc.pth'))
            else:
                curr_patience -= 1

            if curr_patience <= 0:
                break

            # Display some statistics
            #logger.info(f'{epoch},{train_loss},{val_loss}')

    def test_net(self, dl):
        # Load the best weights
        if WEIGHT_DIR.joinpath('classifier_fc.pth').exists():
            self.load_state_dict(torch.load(WEIGHT_DIR.joinpath('classifier_fc.pth'), weights_only=True))

        # Compute the prediction for the whole dataset
        self.eval()
        all_preds = None
        all_labels = None
        with torch.no_grad():
            for batch, labels in dl:
                batch = batch.to(DEVICE)
                preds = self(batch)

                if all_preds is None:
                    all_preds = preds
                    all_labels = labels
                else:
                    all_preds = torch.cat([all_preds, preds], dim=0)
                    all_labels = torch.cat([all_labels, labels], dim=0)

        # Get a classification report
        all_preds = all_preds.cpu().numpy()
        for idx, pred in enumerate(all_preds):
            idx_max = np.argmax(pred)
            all_preds[idx] = np.zeros_like(pred)
            all_preds[idx][idx_max] = 1
        all_labels = all_labels.cpu().numpy()

        #print(classification_report(all_labels, all_preds))
        logger.info(f'Classification accuracy: {accuracy_score(all_labels, all_preds)}')
