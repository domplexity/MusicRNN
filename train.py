import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.autograd import Variable
import argparse
from tqdm import tqdm

from model import *


class MusicDataset(Dataset):
    def __init__(self, file, sequence_length=70):
        self.tensor = torch.LongTensor(np.load(file))
        self.sequence_length = sequence_length

    def __len__(self):
        return self.tensor.size(0) // self.sequence_length

    def __getitem__(self, idx):
        # the returned tensor is of form (input, target)
        return (self.tensor[idx*self.sequence_length:(idx+1)*self.sequence_length],
                self.tensor[idx*self.sequence_length+1:(idx+1)*self.sequence_length+1])

# main
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--directory', type=str)
    argparser.add_argument('--loadWeights', action='store_true')
    args = argparser.parse_args()

    print("About to start training with directory %s, loadWeights %s" % (args.directory, args.loadWeights))

    # hyperparameters
    hidden_size = 100
    n_layers = 1
    batch_size = 10
    n_epochs = 2000
    vocabulary_size = 277

    # load weights if specified
    if args.loadWeights:
        model = torch.load("weights.pth")
        print("loaded weights")
    else:
        model = MidiRNN(vocabulary_size, hidden_size, vocabulary_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # closure for training
    def train(input, target):
        hidden = model.init_hidden(batch_size)
        model.zero_grad()
        loss = 0

        for c in range(input.size(1)):
            output, hidden = model(input[:, c], hidden)
            loss += criterion(output.view(batch_size, -1), target[:, c])

        loss.backward()
        optimizer.step()

        return loss.data[0] / input.size()[0]

    # load dataset
    dataset = MusicDataset(file=args.directory + "/corpus.npy", sequence_length=200)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)

    try:
        print("training for %d epochs..." % n_epochs)

        total_minibatches = len(dataset)
        loss_total = 0
        all_losses = []

        for epoch in tqdm(range(1, n_epochs + 1)):
            for num_of_minibatch, (input, target) in enumerate(dataloader):
                input_data = Variable(input)
                target_data = Variable(target)

                # calculate loss
                loss = train(input_data, target_data)
                loss_total += loss
                all_losses.append(loss)

                # logging
                if num_of_minibatch % 1:
                    np.save('training_log', all_losses)
                    print("minibatch %d of %d has loss %.4f" % (num_of_minibatch, total_minibatches // batch_size, loss))

        print("Saving...")
        torch.save(decoder, 'weights.pth')

    except KeyboardInterrupt:
        print("Saving before quit...")
        torch.save(decoder, 'weights.pth')
        np.save('training_log', all_losses)

