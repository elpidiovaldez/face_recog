#!/usr/bin/python3

# pytorch mlp for multiclass classification
# See https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

import sys
import argparse
import numpy
import torch
from classifier_model import MLP
from numpy import vstack
from numpy import argmax
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import SGD
from torch.nn import CrossEntropyLoss


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        df = torch.load(str(path.joinpath('embeddings.pt')))
        arr = numpy.array(df, dtype=object)
        self.X = torch.from_numpy(numpy.vstack(arr[:, 2]).astype('float32'))
        self.Y = arr[:, 0].astype('str')
        # label encode target 
        self.labelEncoder = LabelEncoder()
        self.Y = self.labelEncoder.fit_transform(self.Y)
        self.n_inputs = self.X.size(1)
        self.n_outputs = self.labelEncoder.classes_.size
        print(self.n_inputs, self.n_outputs)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(500):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()


# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        inputs = inputs.to(device)
        targets = targets.to(device)
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().cpu().numpy()
        actual = targets.detach().cpu().numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc


def make_classifier(args):
    data_dir = Path(args.data).resolve()
    aligned_dir = data_dir.joinpath('photos_aligned_faces')

    print('Running on ', device)
    # prepare the data

    # load the dataset
    dataset = CSVDataset(data_dir)
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)

    print('Training set = ', len(train_dl.dataset), 'Test set =', len(test_dl.dataset))
    # define the network
    model = MLP(dataset.n_inputs, dataset.n_outputs, dataset.labelEncoder).to(device)
    # train the model
    train_model(train_dl, model)
    # evaluate the model
    acc = evaluate_model(test_dl, model)
    print('Accuracy on Test Set: %.3f' % acc)

    save_path = str(data_dir.joinpath('face_classifier.pt'))
    with open(save_path, 'wb') as pickle_file:
        torch.save(model, pickle_file)

    print('Saved classifier as: "{}"'.format(save_path))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str,
                        default='../data',
                        help='Path to the data folder (which contains the photos_raw folder)')

    return parser.parse_args(argv)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    make_classifier(parse_arguments(sys.argv[1:]))
