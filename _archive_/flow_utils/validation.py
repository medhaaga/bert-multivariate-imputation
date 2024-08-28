import math
import numpy as np
import torch


def val_maf(model, val_loader):
    model.eval()
    val_loss = []

    for batch in val_loader:
        u, log_det = model.forward(batch.float())
        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= log_det
        val_loss.extend(negloglik_loss.tolist())

    N = len(val_loader.dataset)
    loss = np.sum(val_loss) / N
    return loss


def val_made(model, val_loader):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch in val_loader:
            out = model.forward(batch.float())
            mu, logp = torch.chunk(out, 2, dim=1)
            u = (batch - mu) * torch.exp(0.5 * logp)

            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
            negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
            negloglik_loss -= 0.5 * torch.sum(logp, dim=1)
            negloglik_loss = torch.mean(negloglik_loss)

            val_loss.append(negloglik_loss)

    N = len(val_loader)
    loss = np.sum(val_loss) / N
    return loss