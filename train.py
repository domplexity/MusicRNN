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
        # output tensor is shifted by one note to the right with respect to input tensor
        input_tensor = self.tensor[idx * self.sequence_length:(idx + 1) * self.sequence_length]
        output_tensor = self.tensor[idx * self.sequence_length + 1:(idx + 1) * self.sequence_length + 1]

        # the returned tensor is of form (input, target)
        return (input_tensor, output_tensor)

# main
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--directory', type=str)
    argparser.add_argument('--loadWeights', action='store_true')
    argparser.add_argument('--useGpu', action='store_true')
    argparser.add_argument('--clipGradients', action='store_true')
    args = argparser.parse_args()

    print("About to start training with directory %s, loadWeights %s" % (args.directory, args.loadWeights))

    # hyperparameters
    hidden_size = 100
    n_layers = 1
    batch_size = 10
    n_epochs = 2000
    vocabulary_size = 285  # 88 note-on and note-off events, 101 DtEvents, 8 VelocityEvents

    # load weights if specified
    if args.loadWeights:
        model = torch.load("weights.pth")
        print("loaded weights")
    else:
        model = MidiRNN(vocabulary_size, hidden_size, vocabulary_size)

    # ensure that we run on gpu if specified
    use_gpu = args.useGpu
    if use_gpu:
        model.cuda()

    # loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    # closure for training
    def train(input, target):
        # initialize hidden state
        hidden = model.init_hidden(batch_size)

        # initalize gradients and loss
        model.zero_grad()
        loss = 0

        # convert to gpu if needed
        if use_gpu:
            input = input.cuda()
            target = target.cuda()
            hidden = hidden.cuda()

        for c in range(input.size(1)):
            output, hidden = model(input[:, c], hidden)
            loss += criterion(output.view(batch_size, -1), target[:, c])

        loss.backward()

        if args.clipGradients:
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        return loss.data[0] / input.size()[0]

    # load dataset
    dataset = MusicDataset(file=args.directory + "/corpus.npy", sequence_length=200)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

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
                if num_of_minibatch % 10:
                    np.save('training_log', all_losses)
                    print("minibatch %d of %d has loss %.4f" % (num_of_minibatch, total_minibatches // batch_size - 1, loss))

        print("Saving...")
        torch.save(model, 'weights.pth')

    except KeyboardInterrupt:
        print("Saving before quit...")
        torch.save(model, 'weights.pth')
        np.save('training_log', all_losses)

