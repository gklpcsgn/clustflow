"""
Creates a flexible attention-based embedding layer that can handle multiple categorical features with different cardinalities and optional continuous features.

Accepts a list of cardinalities and embedding dimensions for each categorical feature, and can optionally include continuous features. The module uses a transformer encoder to process the embeddings, allowing for flexible attention mechanisms.

Example
-------
>>> from clustflow.embedding.attention_embedding import FlexibleAttentionEmbedding
>>> cardinals = [10, 20, 30]  # Cardinalities of categorical features
>>> emb_dims = [8, 16, 32]  # Embedding dimensions for each categorical feature
>>> cont_dim = 5  # Number of continuous features
>>> model = FlexibleAttentionEmbedding(cardinals, emb_dims, cont_dim, embed_dim=64, n_heads=4, depth=2, dropout=0.1)
>>> x_cat = torch.randint(0, 10, (32, 3))  # Example categorical input
>>> x_cont = torch.randn(32, 5)  # Example continuous input
>>> embedding = model(x_cat, x_cont)  # Forward pass to get embeddings
"""

import torch
import torch.nn as nn

class FlexibleAttentionEmbedding(nn.Module):
    """
    Creates a flexible attention-based embedding layer for categorical and continuous features.

    Parameters:
    ----------
    cardinals : list
        List of cardinalities for each categorical feature.
    emb_dims : list
        List of embedding dimensions for each categorical feature.
    cont_dim : int
        Number of continuous features. If 0, no continuous features are used.
    embed_dim : int
        Output embedding dimension.
    n_heads : int
        Number of attention heads.
    depth : int
        Number of transformer encoder layers.
    dropout : float
        Dropout rate for the transformer layers.
    with_decoder : bool, optional
        If True, includes a decoder for reconstructing the input features. Default is False.
    output_dim : int, optional
        If `with_decoder` is True, the dimension of the output from the decoder.

    Returns:
    -------
    embedding : torch.Tensor
        The final embedding output after processing through the transformer layers.
    reconstruction : torch.Tensor, optional
        If `with_decoder` is True, returns the reconstructed input features.
    """
    def __init__(self, cardinals, emb_dims, cont_dim, embed_dim=64, n_heads=4, depth=2, dropout=0.1, with_decoder=False, output_dim=None):
        super().__init__()

        # Validate input dimensions
        assert len(cardinals) == len(emb_dims), "Cardinalities and embedding dimensions must match in length."

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(cardinals, emb_dims)
        ])

        # If continuous features are used, project them to the sum of embedding dimensions
        self.cont_proj = nn.Linear(cont_dim, sum(emb_dims)) if cont_dim > 0 else None

        # Project the concatenated embeddings to the final embedding dimension
        total_input_dim = sum(emb_dims)*2 if cont_dim > 0 else sum(emb_dims)
        self.token_proj = nn.Linear(total_input_dim, embed_dim)

        # Create transformer encoder layers
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dropout=dropout, batch_first=True)
            for _ in range(depth)
        ])

        # Final output projection layer
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Optional decoder for reconstruction
        self.with_decoder = with_decoder
        if with_decoder:
            if output_dim is None:
                raise ValueError("output_dim must be provided if with_decoder=True")
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, output_dim)
            )

    def forward(self, x_cat, x_cont=None):
        cat_embed = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)

        if self.cont_proj is not None and x_cont is not None:
            cont_embed = self.cont_proj(x_cont)
            tokens = torch.stack([cat_embed, cont_embed], dim=1)  # [B, 2, D]
        else:
            tokens = cat_embed.unsqueeze(1)  # [B, 1, D]

        x = self.token_proj(tokens)

        for block in self.blocks:
            x = block(x)

        pooled = x.mean(dim=1)
        embedding = self.output_proj(pooled)

        if self.with_decoder:
            reconstruction = self.decoder(embedding)
            return embedding, reconstruction

        return embedding
