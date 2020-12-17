import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


# Reconstruction error + KL divergence
def loss_function(X_rec, X, norm, weights, num_nodes, mu, log_var):

    # Reconstruction error (with logits)
    BCE = norm * F.binary_cross_entropy(X_rec, X, weight=weights, reduction="mean")

    # KL divergence
    KLD = (.5 / num_nodes) * (1 + 2 * log_var - mu ** 2 - log_var.exp() ** 2).sum(1).mean()

    return BCE - KLD


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
