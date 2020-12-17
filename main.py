from graph_loader import GraphLoader
from vgae import VGAE
from training_and_validation import *
import torch.optim as optim

# Default device
DEVICE = "cpu"

# Model hidden sizes
HIDDEN_SIZE = 32
LATENT_SIZE = 16

# Training epochs
EPOCHS = 200

if __name__ == "__main__":

    # Load a dataset
    loader = GraphLoader("cora", DEVICE)

    # Split the data, get masked edges to perform validation and compute loss coefficients
    X, As, Es, coef = loader.get_train_val_test_data()

    # Unpack everything in a human readable syntax
    A_norm, A_label, A_backup = As
    E_train_t, E_val_t, E_val_f, E_test_t, E_test_f = Es
    norm, weights = coef

    # Get shapes from the data to initialize the model
    n, m, o = loader.get_features_size()

    # Initialize a new VGAE (without dropout)
    vgae = VGAE(A_norm, n, m, o, HIDDEN_SIZE, LATENT_SIZE)
    vgae.to(DEVICE)

    # Train for 200 epochs
    optimizer = optim.Adam(vgae.parameters(), lr=0.01)
    for epoch in range(1, EPOCHS + 1):

        # Make training step
        loss = train(vgae, optimizer, X, A_label, norm, weights, n)
        print(f"Epoch {epoch}, Loss: {loss}")

        # Every V epochs validate the model
        if epoch % 10 == 0:

            auc, ap = test(vgae, E_test_t, E_test_f)
            print(f"Epoch {epoch}, AUC: {auc} AP: {ap}")
