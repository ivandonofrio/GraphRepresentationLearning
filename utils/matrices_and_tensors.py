import scipy.sparse as sp
import numpy as np
import torch


def sparse_to_tuple(matrix):

    # Cast to COO matrix
    if not sp.isspmatrix_coo(matrix):
        matrix = matrix.tocoo()

    # Get data from sparse matrix
    coords = np.vstack((matrix.row, matrix.col)).transpose()
    values = matrix.data
    shape = matrix.shape

    return coords, values, shape


def normalize_adjacency_matrix(A):

    # Add identity matrix to avoid zero diagonal
    A = sp.coo_matrix(A)
    A_hat = A + sp.eye(A.shape[0])

    # Normalize adjacency matrix using rows degree matrix inverse squared
    D_hat = sp.diags(np.power(np.array(A_hat.sum(1)), -0.5).flatten())
    A_norm = A_hat.dot(D_hat).transpose().dot(D_hat).tocoo()

    return A_norm


def sparse_to_tensor(X, device="cpu"):

    # Expand input tuple
    coords, values, shape = X

    return torch.sparse.FloatTensor(
        torch.LongTensor(coords).t(),
        torch.FloatTensor(values),
        torch.Size(shape)
    ).to(device)
