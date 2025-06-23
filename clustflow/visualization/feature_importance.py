"""
Visualizes top-ranked features based on importance scores.

Use with output from `compute_feature_importance()`.

Example
-------
>>> from clustflow.visualization.feature_importance import plot_top_features
>>> plot_top_features(importance['numerical'], title="Top Numerical Features")
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_top_features(importance_df, title="Top Feature Importances", top_n=10):
    """
    Plots bar chart of top N important features.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'score' columns (e.g., from compute_feature_importance).
    title : str, default="Top Feature Importances"
        Title of the plot.
    top_n : int, default=10
        Number of top features to display.

    Returns
    -------
    None
    """
    df = importance_df.sort_values(by='score', ascending=False).head(top_n)
    plt.figure(figsize=(8, 5))
    sns.barplot(x='score', y='feature', data=df, palette='viridis')
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
