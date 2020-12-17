import torch
import torch.nn as nn


class SparseDropout(torch.nn.Module):

    def __init__(self, dropout=0):
        super(SparseDropout, self).__init__()

        # Store dropout probability
        self.dropout = dropout

    def forward(self, X):
        # Store X sizes
        size = X._values().size()

        # Generate dropout random mask
        mask = (torch.rand(size) + (1 - self.dropout)).floor().type(torch.bool)
        indices = X._indices()[:, mask]
        values = X._values()[mask] * (1.0 / (1 - self.dropout))

        return torch.sparse.FloatTensor(
            indices,
            values,
            X.shape
        )


class GCNLayer(nn.Module):

    def __init__(self, m, h, sigma, is_sparse=False):
        super().__init__()

        # Store activation function
        self.sigma = sigma

        # Store sparse flag
        self.is_sparse = is_sparse

        # Initialize weights with Xavier method
        W = torch.empty(m, h)
        self.W = nn.Parameter(nn.init.xavier_uniform_(W))

    def forward(self, A_hat, X):

        # If we are working with sparse matrices use optimized methods
        if self.is_sparse:
            Z = torch.sparse.mm(X, self.W)

        # Otherwise use the standard ones
        else:
            Z = torch.mm(X, self.W)

        return self.sigma(torch.sparse.mm(A_hat, Z))


class InnerProductDecoder(nn.Module):

    def __init__(self, sigma):
        super().__init__()

        # Store activation function
        self.sigma = sigma

    def forward(self, Z):
        return self.sigma(torch.matmul(Z, Z.t()))
