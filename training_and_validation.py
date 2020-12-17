import torch
from loss_and_metrics import *


def train(model, optimizer, X, A, norm, weights, n):

    # Set the VGAE to be trained
    model.train()

    # Zero grad
    optimizer.zero_grad()

    # Compute, collect loss and make optimization step
    A_rec, mu, log_var = model(X)
    A_rec_vec = A_rec.view(-1)
    A_vec = A.to_dense().view(-1)

    # BCE + KLD loss
    loss = loss_function(A_rec_vec, A_vec, norm, weights, n, mu, log_var)

    # Optimization step
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, E_t, E_f):
    with torch.no_grad():

        # Set the VGAE to be evaluated
        model.eval()

        return get_auc_and_ap(model, E_t, E_f)
