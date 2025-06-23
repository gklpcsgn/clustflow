
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def scatter_2d(X, labels, color_by=None, title="", save_path=None):
    """
    X: 2D array (PCA or UMAP)
    labels: cluster assignments
    color_by: optional array (e.g., true labels)
    """
    df = pd.DataFrame(X, columns=["dim1", "dim2"])
    df["cluster"] = labels
    if color_by is not None:
        df["color"] = color_by
        hue = "color"
    else:
        hue = "cluster"

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="dim1", y="dim2", hue=hue, palette="tab10", s=50, alpha=0.8, edgecolor="black")
    plt.title(title or "Cluster Visualization")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
