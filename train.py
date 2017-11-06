from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import math
from model import *


class MusicDataset(Dataset):
    def __init__(self, file, sequence_length=70):
        self.tensor = torch.LongTensor(np.load(file))
        self.sequence_length = sequence_length

    def __len__(self):
        return self.tensor.size(0) // self.sequence_length

    def __getitem__(self, idx):
        # target is shifted by one note to the right with respect to input
        input_tensor = self.tensor[idx * self.sequence_length:(idx + 1) * self.sequence_length]
        target_tensor = self.tensor[idx * self.sequence_length + 1:(idx + 1) * self.sequence_length + 1]

        # the returned tensor is of form (input, target)
        return (input_tensor, target_tensor)


def save_model(model, use_gpu):
    torch.save(model, 'weights.pth')
    # save also cpu version of model in case we are training on gpu
    if use_gpu:
        torch.save(model.cpu(), 'weights_cpu.pth')

# main
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--directory', type=str)
    argparser.add_argument('--loadWeights', action='store_true')
    argparser.add_argument('--useGpu', action='store_true')
    argparser.add_argument('--clipGradients', action='store_true')
    argparser.add_argument('--batch_size', type=int, default=100)
    argparser.add_argument('--n_epochs', type=int, default=2000)
    argparser.add_argument('--lr', type=float, default=0.0005)
    argparser.add_argument('--seq_len', type=int, default=200)
    argparser.add_argument('--eval_every_n_epoch', type=int, default=1)
    args = argparser.parse_args()

    print("About to start training with directory %s, loadWeights %s" % (args.directory, args.loadWeights))

    # hyperparameters
    hidden_size = 200
    n_layers = 4
    batch_size = args.batch_size
    seq_len = args.seq_len
    n_epochs = args.n_epochs
    vocabulary_size = 285  # 88 note-on and note-off events, 101 DtEvents, 8 VelocityEvents

    print("Architecture: (%s layers,  %s hidden units, %s vocabulary size)" % (n_layers, hidden_size, vocabulary_size))

    # load weights if specified
    if args.loadWeights:
        model = torch.load("weights.pth")
        print("loaded weights")
    else:
        model = MidiRNN(vocabulary_size, hidden_size, vocabulary_size, n_layers=n_layers)

    # ensure that we run on gpu if specified
    use_gpu = args.useGpu
    if use_gpu:
        model.cuda()
        print("Will run on GPU.")

    # loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
        # iterate over range with length of sequence_length
        for c in range(input.size(1)):
            output, hidden = model(input[:, c], hidden)
            loss += criterion(output.view(batch_size, -1), target[:, c])

        loss.backward()

        if args.clipGradients:
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        return loss.data[0] / input.size(1)  # normalize loss by sequence_length


    # closure for evaluation
    def evaluate():
        # initialize hidden state
        hidden = model.init_hidden(batch_size)

        total_loss = 0

        for input, target in test_data_loader:
            input = Variable(input)
            target = Variable(target)

            # convert to gpu if needed
            if use_gpu:
                input = input.cuda()
                target = target.cuda()
                hidden = hidden.cuda()

            for c in range(input.size(1)):
                output, hidden = model(input[:, c], hidden)
                total_loss += criterion(output.view(batch_size, -1), target[:, c])

        # normalize loss
        total_loss = total_loss.data[0] / len(test_set)

        return total_loss, math.exp(total_loss)


    # load datasets
    training_set = MusicDataset(file=args.directory + "/corpus.npy", sequence_length=seq_len)
    train_data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    test_set = MusicDataset(file=args.directory + "/corpus_test.npy", sequence_length=seq_len)
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    try:
        print("training for %d epochs..." % n_epochs)

        total_minibatches = len(training_set)
        loss_total = 0
        all_losses = []
        all_perplexities = []

        for epoch in tqdm(range(1, n_epochs + 1)):
            for num_of_minibatch, (input, target) in enumerate(train_data_loader):
                input_data = Variable(input)
                target_data = Variable(target)

                # calculate loss
                loss = train(input_data, target_data)
                loss_total += loss
                all_losses.append(loss)

                # logging
                if num_of_minibatch % 10 == 0:
                    np.save('training_log', all_losses)
                    print("minibatch %d of %d has loss %.4f" % (num_of_minibatch, total_minibatches // batch_size - 1, loss))


            # evaluate test set
            if epoch % args.eval_every_n_epoch == 0:
                loss_test, perplexity_test = evaluate()
                all_perplexities.append(perplexity_test)
                np.save('test_log', all_perplexities)
                print("Evaluating test set: loss %.4f and perplexity %.4f" % (loss_test, perplexity_test))

        print("Saving...")
        save_model(model, use_gpu)

    except KeyboardInterrupt:
        print("Saving before quit...")
        save_model(model, use_gpu)
        np.save('training_log', all_losses)
        np.save('test_log', all_perplexities)