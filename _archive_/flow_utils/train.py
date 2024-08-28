import datetime
import math
import numpy as np
import torch

# utility functions
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train_one_epoch_maf(model, optimizer, train_loader):

    model.train()
    train_loss = 0
    for batch in train_loader:
        u, log_det = model.forward(batch.float())

        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= log_det
        negloglik_loss = torch.mean(negloglik_loss)

        negloglik_loss.backward()
        train_loss += negloglik_loss.item()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = np.sum(train_loss) / len(train_loader)
    return avg_loss


def train_one_epoch_made(model, optimizer, train_loader):
    model.train()
    train_loss = 0
    for batch in train_loader:

        out = model.forward(batch.float())
        mu, logp = torch.chunk(out, 2, dim=1)
        u = (batch - mu) * torch.exp(0.5 * logp)

        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= 0.5 * torch.sum(logp, dim=1)

        negloglik_loss = torch.mean(negloglik_loss)

        negloglik_loss.backward()
        train_loss += negloglik_loss.item()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = np.sum(train_loss) / len(train_loader)
    return avg_loss