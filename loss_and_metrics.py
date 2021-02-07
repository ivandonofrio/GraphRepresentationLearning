import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


# Reconstruction error + KL divergence
def loss_function(A_rec, A, norm, weights, num_nodes, mu, log_var):

    # Reconstruction error (with logits)
    BCE = norm * F.binary_cross_entropy(A_rec, A, weight=weights, reduction="mean")

    # KL divergence
    KLD = (.5 / num_nodes) * (1 + 2 * log_var - mu ** 2 - log_var.exp() ** 2).sum(1).mean()

    return BCE - KLD


# Reconstruction error for anomaly detection, a bit naïve
def loss_function_naive_anomaly_detection(X_rec, A_rec, X, A, alpha, mu, log_var):

    # Reconstruction errors and loss

    #A_reconstruction_errors = torch.sqrt(torch.sum(torch.square(A_rec - A), 1))
    #MSE_A = torch.mean(A_reconstruction_errors)

    #X_reconstruction_errors = torch.sqrt(torch.sum(torch.square(X_rec - X), 1))
    #MSE_X = torch.mean(X_reconstruction_errors)

    A_reconstruction_errors = (A_rec - A).square().sum(1).sqrt()
    MSE_A = A_reconstruction_errors.mean()

    X_reconstruction_errors = (X_rec - X).square().sum(1).sqrt()
    MSE_X = X_reconstruction_errors.mean()

    MSE = alpha * MSE_X + (1 - alpha) * MSE_A

    # KL divergence
    KLD = (.5 / num_nodes) * (1 + 2 * log_var - mu ** 2 - log_var.exp() ** 2).sum(1).mean()

    return (MSE - KLD), A_reconstruction_errors, X_reconstruction_errorsì


def get_auc_and_ap(model, E_t, E_f):

    # Who wants another sigmoid?
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Generate reconstructed adjacency matrix from the last Z
    A_rec = model.reconstruct_from_latent_space().cpu().numpy()

    # Positive and negative predictions
    positive_pred = [sigmoid(A_rec[r, c]) for r, c in E_t]
    negative_pred = [sigmoid(A_rec[r, c]) for r, c in E_f]

    # Generate labels and prediction sets to apply metrics
    predictions = np.hstack([positive_pred, negative_pred])
    labels = np.hstack([np.ones(len(positive_pred)), np.zeros(len(negative_pred))])

    return roc_auc_score(labels, predictions), average_precision_score(labels, predictions)
