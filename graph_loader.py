import torch
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.io
import scipy.sparse as sp

from utils.edge_mask import mask_test_edges
from utils.matrices_and_tensors import sparse_to_tuple, sparse_to_tensor, normalize_adjacency_matrix


class GraphLoader:

    def __init__(self, dataset="cora", task="representation_learning", device="cpu"):

        # Store dataset name and device on which load the data
        self.dataset = dataset
        self.device = device
        self.task = task

        # Load adjacency matrix and features as tuples
        A, X, labels = self.load_graph()

        self.A = A
        self.X = sparse_to_tuple(X)
        self.labels = labels

    def load_graph(self):

        # Load the data: x, tx, allx, graph
        objects = []

        # Load dataset for Representation Learning experiments
        if self.task == "representation_learning":

            for ext in ["x", "tx", "allx", "graph", "labels"]:
                with open(f"graph_data/{self.task}/ind.{self.dataset}.{ext}", 'rb') as f:
                    objects.append(pkl.load(f, encoding='latin1'))

            x, tx, allx, graph, labels = tuple(objects)

            # Load test set indices
            test_idx_reorder = [int(line.strip()) for line in open(f"graph_data/{self.task}/ind.{self.dataset}.test.index")]
            test_idx_range = np.sort(test_idx_reorder)

            # Fix citeseer dataset (there are some isolated nodes in the graph)
            if self.dataset == "citeseer":

                # Find isolated nodes, add them as zero-vecs into the right position
                test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
                tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
                tx_extended[test_idx_range - min(test_idx_range), :] = tx
                tx = tx_extended

            # Get adjacency matrix and graph features
            X = sp.vstack((allx, tx)).tolil()
            X[test_idx_reorder, :] = X[test_idx_range, :]
            A = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        # Load data for Anomaly Detection experiments
        elif self.task == "anomaly_detection":

            # Load the dataset
            data = scipy.io.loadmat(f"graph_data/{self.task}/{self.dataset}.mat")

            # Unfortunately the labels weren't chose by me...
            X = sp.csr_matrix(data["X"])
            A = sp.lil_matrix(data["A"])
            labels = data["gnd"].squeeze()

        return A, X, labels

    def get_features_size(self):

        # Extract dimensions from the features matrix to initialize the model
        n = self.X[2][0]
        m = self.X[2][1]
        o = self.X[1].shape[0]

        return n, m, o

    def get_train_val_test_data(self):

        # Generate tensor from features matrix X
        X = sparse_to_tensor(self.X, device=self.device)

        # Get training, validation and training data masking edges (t: true, f: false) to be predicted
        if self.task == "representation_learning":
            A_train, E_train_t, E_val_t, E_val_f, E_test_t, E_test_f = mask_test_edges(self.A)
        else:
            A_train = E_train_t = E_val_t = E_val_f = E_test_t = E_test_f = None
            A_train = self.A.copy()

        # Preserve initial adjacency matrix (without zeroes in diag) to evaluate reconstruction error
        A_backup = self.A - sp.dia_matrix((self.A.diagonal()[np.newaxis, :], [0]), shape=self.A.shape)
        A_backup.eliminate_zeros()

        # Normalize A_train as described in the GCN paper to optimize training
        A_norm = sparse_to_tuple(normalize_adjacency_matrix(A_train))
        A_norm = sparse_to_tensor(A_norm, self.device)

        # Generate identity matrix to A_train to evaluate loss function
        A_label = sparse_to_tuple(A_train + sp.eye(A_train.shape[0]))
        A_label = sparse_to_tensor(A_label, self.device)

        # Compute norm and pos_weight factors
        pos_weight = float(A_train.shape[0] ** 2 - A_train.sum()) / A_train.sum()
        norm = A_train.shape[0] ** 2 / (float(A_train.shape[0] ** 2 - A_train.sum()) * 2)

        # Generate weight mask for BCE loss evaluation
        weight_mask = A_label.to_dense().view(-1) == 1
        weights = torch.ones(weight_mask.size(0))
        weights[weight_mask] = pos_weight
        weights = weights.to(self.device)

        return (
            X,
            (A_norm, A_label, A_backup),
            (E_train_t, E_val_t, E_val_f, E_test_t, E_test_f),
            (norm, weights)
        )
