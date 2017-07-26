import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import numpy as np


from model import *
import preprocessing


def generate(model, initialization_sequence, predict_len=1000, temperature=0.9):
    hidden = model.init_hidden(1)
    prime_input = Variable(initialization_sequence.unsqueeze(0))

    predicted = initialization_sequence.tolist()

    # build up hidden state
    for p in range(len(initialization_sequence) - 1):
        _, hidden = model(prime_input[:, p], hidden)

    input = prime_input[:, -1]

    for p in range(predict_len):
        output, hidden = model(input, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        num_of_predicted_midi_event = torch.multinomial(output_dist, 1)[0]

        # Add predicted midi event and use as next input
        predicted.append(num_of_predicted_midi_event)
        input = Variable(torch.LongTensor([num_of_predicted_midi_event]).unsqueeze(0))

    return predicted


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    args = argparser.parse_args()

    model = torch.load(args.filename)
    initialization_sequence = torch.LongTensor(np.load("beeth/tensors/corpus.npy")[:100])

    midi_numbers = generate(model, initialization_sequence)
    preprocessing.write_events(midi_numbers, "output.mid")