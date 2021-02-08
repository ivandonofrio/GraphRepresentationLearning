import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, recall_score


# Reconstruction error + KL divergence
def loss_function(A_rec, A, norm, weights, num_nodes, mu, log_var):

    # Reconstruction error (with logits)
    BCE = norm * F.binary_cross_entropy(A_rec, A, weight=weights, reduction="mean")

    # KL divergence
    KLD = (.5 / num_nodes) * (1 + 2 * log_var - mu ** 2 - log_var.exp() ** 2).sum(1).mean()

    return BCE - KLD


# Reconstruction error for anomaly detection, a bit naïve
def loss_function_naive_anomaly_detection(X_rec, A_rec, X, A, alpha, num_nodes, mu, log_var):

    # Reconstruction errors and loss
    A_clone = A_rec.clone().detach()
    X_clone = X_rec.clone().detach()

    A_reconstruction_errors = (A_clone - A).square().sum(1).sqrt()
    X_reconstruction_errors = (X_clone - X).square().sum(1).sqrt()

    MSE_A = (A_rec - A).square().sum(1).sqrt().mean()
    MSE_X = (X_rec - X).square().sum(1).sqrt().mean()

    MSE = alpha * MSE_X + (1 - alpha) * MSE_A
    reconstruction_errors = X_reconstruction_errors.mul(alpha) + A_reconstruction_errors.mul(1 - alpha)

    # KL divergence
    KLD = (.5 / num_nodes) * (1 + 2 * log_var - mu ** 2 - log_var.exp() ** 2).sum(1).mean()

    return (MSE - KLD), reconstruction_errors


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


def get_k_accuracy_and_recall(y_pred):

    # Generate an array of ones to compare the predictions
    y_expected = np.ones_like(y_pred)

    return accuracy_score(y_expected, y_pred), recall_score(y_expected, y_pred)


def get_auc_for_anomaly_detection(y_pred, y_true):
    return roc_auc_score(y_true, y_pred)


