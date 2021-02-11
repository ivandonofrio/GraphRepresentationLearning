from layers import *


class VGAE_features(nn.Module):

    def __init__(self, A, n, m, o, h, z, dropout=0):
        super(VGAE_features, self).__init__()

        # Store adjacency matrix
        self.A = A

        # Placeholder for last latent variable
        self.mu = None
        self.log_var = None
        self.Z = None

        # Store model sizes
        self.n = n
        self.m = m
        self.o = o
        self.h = h
        self.z = z

        # Store dropout
        self.dropout = dropout

        # Model layers
        # Encoder layers
        self.encoder_dropout_sparse = SparseDropout(self.dropout)
        self.encoder_conv = GCNLayer(m, h, nn.ReLU(), is_sparse=True)
        self.encoder_dropout_dense = nn.Dropout(self.dropout)
        self.encoder_mean = GCNLayer(h, z, nn.Identity())
        self.encoder_var = GCNLayer(h, z, nn.Identity())

        # Adjacency Decoder layers
        self.decoder_adj_dropout_dense = nn.Dropout(self.dropout)
        self.decoder_adj_inner_product = InnerProductDecoder(nn.Sigmoid())

        # Features Decoder layers
        self.decoder_feat_conv = GCNLayer(z, m, nn.ReLU())

    # VAE architecture
    def reparametrization(self, mu, log_var):

        # Reparametrization trick
        var = torch.exp(log_var)
        eps = torch.randn_like(var)

        return eps.mul(var).add_(mu)

    def encoder(self, X):

        # Dropout input matrix
        X = self.encoder_dropout_sparse(X)

        # First convolution layer
        h1 = self.encoder_conv(self.A, X)
        h1 = self.encoder_dropout_dense(h1)

        return self.encoder_mean(self.A, h1), self.encoder_var(self.A, h1)

    def decoder(self, Z):

        # Reconstructed adjacency matrix
        if self.training:
            Z = self.decoder_adj_dropout_dense(Z)

        return self.decoder_adj_inner_product(Z), self.decoder_feat_conv(self.A, Z)

    def reconstruct_from_latent_space(self):

        return self.decoder(self.Z)

    def forward(self, X):

        # Encode data to mean and log var in the latent space
        mu, log_var = self.encoder(X)

        # Perform reparametrization trick and store last latent variable
        self.mu = mu
        self.log_var = log_var
        self.Z = self.reparametrization(mu, log_var)

        # Compute reconstructed adjacency matrix and features
        A_hat, X_hat = self.decoder(self.Z)

        return A_hat, X_hat, mu, log_var
