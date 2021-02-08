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


def train_anomaly_detection(model, optimizer, X, A, n):

    # Set the VGAE to be trained
    model.train()

    # Zero grad
    optimizer.zero_grad()

    # Compute, collect loss and make optimization step
    A_rec, X_rec, mu, log_var = model(X)

    # MSE + KLD loss
    loss, reconstruction_errors = loss_function_naive_anomaly_detection(X_rec, A_rec, X, A, .4, n, mu, log_var)
    local_reconstruction_errors = reconstruction_errors.cpu().numpy()

    # Optimization step
    loss.backward()
    optimizer.step()

    return loss.item(), local_reconstruction_errors


def test(model, E_t, E_f):
    with torch.no_grad():

        # Set the VGAE to be evaluated
        model.eval()

        return get_auc_and_ap(model, E_t, E_f)


def test_anomaly_detection(reconstruction_errors, labels):
    
    # Sort nodes indices by decreasing reconstruction error
    worst_nodes = np.argsort(-reconstruction_errors, axis=0)
    metrics = []

    # Evaluate metrics for different K-worst nodes
    K = [50, 100, 200, 300]

    for k in K:

        # Take the K-worst nodes and compute accuracy and recall
        worst_K_nodes = labels[worst_nodes[:k]]
        accuracy, recall = get_k_accuracy_and_recall(worst_K_nodes)

        metrics.append({'accuracy': accuracy, 'recall': recall})

    return dict(zip(K, metrics))

