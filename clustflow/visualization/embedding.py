"""
Plots embeddings using UMAP or t-SNE.

Accepts either a numpy array or a PyTorch tensor of embeddings.

Example
-------
>>> from clustflow.visualization.embedding import plot_embeddings
>>> plot_embeddings(z, labels=labels, method="umap", title="UMAP Embeddings")
>>> plot_embeddings(z, labels=labels, method="tsne", title="t-SNE Embeddings")
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from umap import UMAP

def plot_embeddings(z, labels=None, method="umap", title="Embeddings", figsize=(8, 6), save_path=None):
    """
    Visualize embeddings using UMAP or t-SNE.

    Parameters:
    ----------
    z : np.ndarray or torch.Tensor
        Embeddings (shape: [N, D]).
    labels : np.ndarray or list, optional
        Labels for coloring the plot. If None, no coloring is applied.
    method : str
        Dimensionality reduction method to use ("umap" or "tsne").
    title : str
        Title of the plot.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the plot. If None, the figure object is returned instead of saving.
    ----------

    Returns:
    -------
    matplotlib.figure.Figure: The figure object if save_path is None.
    """
    # Convert tensor to numpy array if necessary
    if hasattr(z, 'detach'):
        z = z.detach().cpu().numpy()

    # Reduce dimensions to 2D
    if method == "umap":
        reducer = UMAP(n_components=2, random_state=42)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Unsupported method. Use 'umap' or 'tsne'.")

    z_2d = reducer.fit_transform(z)

    # Create the plot
    plt.figure(figsize=figsize)
    if labels is not None:
        sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=labels, palette="viridis", s=50)
    else:
        plt.scatter(z_2d[:, 0], z_2d[:, 1], s=50, alpha=0.7)

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        return plt.gcf()
