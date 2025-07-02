"""
Train the FlexibleAttentionEmbedding model using reconstruction loss (MSE).

This module provides a function to train the FlexibleAttentionEmbedding model with categorical and continuous features.

Example
-------
>>> from clustflow.embedding.attention_embedding import FlexibleAttentionEmbedding
>>> cardinals = [10, 20, 30]  # Cardinalities of categorical features
>>> emb_dims = [8, 16, 32]  # Embedding dimensions for each categorical feature
>>> cont_dim = 5  # Number of continuous features
>>> model = FlexibleAttentionEmbedding(cardinals, emb_dims, cont_dim, embed_dim=64, n_heads=4, depth=2, dropout=0.1)
>>> x_cat = torch.randint(0, 10, (32, 3))  # Example categorical input
>>> x_cont = torch.randn(32, 5)  # Example continuous input
>>> trained_model = train_embedding(model, learning_rate=0.001, epochs=10, batch_size=32, x_cat=x_cat, x_cont=x_cont)
>>> embedding = trained_model(x_cat, x_cont)  # Forward pass to get embeddings
>>> print(embedding)
>>> print(embedding.shape)  # Should be [32, 64] if embed_dim is 64
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from clustflow.embedding.attention import FlexibleAttentionEmbedding

def train_embedding(model, learning_rate=0.001, epochs=10, batch_size=32, x_cat=None, x_cont=None):
    """
    Train the FlexibleAttentionEmbedding model using reconstruction loss (MSE).

    Parameters:
    ----------
    model : FlexibleAttentionEmbedding
        Instance of the embedding model.
    learning_rate : float
        Learning rate for the optimizer.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    x_cat : torch.Tensor
        Categorical input data.
    x_cont : torch.Tensor
        Continuous input data.

    Returns:
        model (FlexibleAttentionEmbedding): Trained embedding model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model = model.to(device)

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Move model to device
    model = model.to(device)

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Prepare the data
    if x_cont is not None:
        dataset = TensorDataset(x_cat, x_cont)
    else:
        dataset = TensorDataset(x_cat)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            if x_cont is not None:
                x_cat_batch, x_cont_batch = [data.to(device) for data in batch]
                output = model(x_cat_batch, x_cont_batch)
                target = torch.cat([x_cat_batch, x_cont_batch], dim=1).float()
            else:
                x_cat_batch = batch[0].to(device)
                output = model(x_cat_batch)
                target = x_cat_batch.float()

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    return model
