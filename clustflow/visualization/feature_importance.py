import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_top_features(importance_df, title="Top Feature Importances", top_n=10):
    """
    Plots top N features by score from a DataFrame (from compute_feature_importance).
    """
    df = importance_df.sort_values(by='score', ascending=False).head(top_n)
    plt.figure(figsize=(8, 5))
    sns.barplot(x='score', y='feature', data=df, palette='viridis')
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
