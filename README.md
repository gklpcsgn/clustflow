# ClustFlow: Clustering Toolkit for Mixed-Type Data
clustflow is a modular Python package for clustering real-world datasets — especially those with a mix of numerical and categorical features. It provides built-in support for:

- Multiple imputation strategies (e.g., mean, median, group-median by ZIP)
- Encoding for categorical variables
- PCA & UMAP for dimensionality reduction
- KMeans and Hierarchical clustering
- Comprehensive clustering evaluation
- Feature importance and cluster profiling tools

It is built for **non-technical users in applied domains** such as healthcare, public policy, marketing, and social sciences — with clear APIs and examples for those with limited ML or Python background.
# 🧐 clustflow: Clustering Toolkit for Mixed-Type Data

`` is a modular Python package for clustering real-world datasets — especially those with a mix of numerical and categorical features. It provides built-in support for:

- Multiple imputation strategies (e.g., mean, median, group-median by ZIP)
- Encoding for categorical variables
- PCA & UMAP for dimensionality reduction
- KMeans and Hierarchical clustering
- Comprehensive clustering evaluation
- Feature importance and cluster profiling tools

It is built for **non-technical users in applied domains** such as healthcare, public policy, marketing, and social sciences — with clear APIs and examples for those with limited ML or Python background.

---

## 📦 Installation

```cmd
pip install -r requirements.txt
```

---

## 📘 Example Usage

```python
from clustflow.utils.seed import set_seed
from clustflow.preprocessing.imputer import Imputer
from clustflow.preprocessing.encoder import Encoder
from clustflow.dimensionality.pca import PCAReducer
from clustflow.clustering.kmeans import KMeansCluster
from clustflow.evaluation.metrics import compute_metrics
from clustflow.visualization.scatter import scatter_2d

set_seed(42)

imputer = Imputer(strategy={
    'age': 'mean',
    'income': 'median',
    'neighborhood_clean': ('group_median', 'zip3')
})
df_clean = imputer.fit_transform(df)

encoder = Encoder(strategy='onehot')
X_encoded = encoder.fit_transform(df_clean)

pca = PCAReducer(n_components=2)
X_pca = pca.fit_transform(X_encoded)

kmeans = KMeansCluster(n_clusters=10)
labels = kmeans.fit_predict(X_pca)

metrics = compute_metrics(X_pca, labels)
print(metrics)

scatter_2d(X_pca, labels)
```

---

## 📊 Module Highlights

### `preprocessing.imputer.Imputer`

```python
from clustflow.preprocessing.imputer import Imputer

imputer = Imputer(strategy={
    'age': 'mean',
    'income': ('group_median', 'region')
})
df_filled = imputer.fit_transform(df)
```

---

### `preprocessing.encoder.Encoder`

```python
from clustflow.preprocessing.encoder import Encoder

encoder = Encoder(strategy='onehot')
df_encoded = encoder.fit_transform(df)
```

---

### `dimensionality.pca.PCAReducer`

```python
from clustflow.dimensionality.pca import PCAReducer

pca = PCAReducer(n_components=2)
X_pca = pca.fit_transform(X_encoded)
```

---

### `dimensionality.umap.UMAPReducer`

```python
from clustflow.dimensionality.umap import UMAPReducer

umap = UMAPReducer(n_components=2)
X_umap = umap.fit_transform(X_encoded)
```

---

### `clustering.kmeans.KMeansCluster`

```python
from clustflow.clustering.kmeans import KMeansCluster

kmeans = KMeansCluster(n_clusters=8)
labels = kmeans.fit_predict(X_pca)
```

---

### `clustering.hierarchical.HierarchicalCluster`

```python
from clustflow.clustering.hierarchical import HierarchicalCluster

hc = HierarchicalCluster(linkage_method='complete', affinity='cosine')
labels = hc.fit_predict(X_pca)
```

---

### `evaluation.metrics.compute_metrics`

```python
from clustflow.evaluation.metrics import compute_metrics

results = compute_metrics(X_pca, labels, true_labels=df['target'])
print(results)
```

---

### `evaluation.cluster_feature_importance.compute_feature_importance`

```python
from clustflow.evaluation.cluster_feature_importance import compute_feature_importance

importance = compute_feature_importance(df_encoded, labels, categorical_cols=['gender', 'region'])
```

---

### `evaluation.cluster_summary.summarize_clusters`

```python
from clustflow.evaluation.cluster_summary import summarize_clusters

summary_df = summarize_clusters(df, labels, extra=df['target'], categorical_cols=['gender'])
```

---

### `visualization.scatter.scatter_2d`

```python
from clustflow.visualization.scatter import scatter_2d

scatter_2d(X_pca, labels, color_by=df['target'])
```

---

### `visualization.dendrogram.plot_dendrogram`

```python
from clustflow.visualization.dendrogram import plot_dendrogram

plot_dendrogram(linkage_matrix=model.get_linkage_matrix())
```

---

### `visualization.feature_importance.plot_top_features`

```python
from clustflow.visualization.feature_importance import plot_top_features

plot_top_features(importance['numerical'], title="Top Numerical Features")
```

---

### `visualization.cluster_summary.plot_cluster_heatmap`

```python
from clustflow.visualization.cluster_summary import plot_cluster_heatmap

plot_cluster_heatmap(summary_df)
```

---

## 🚀 Coming Soon

- Deep clustering (DEC / IDEC / DCN)
- Shapley & attention-based feature explanations
- Geographic or ZIP3-based clustering summaries

---

## 💡 Tip for Domain Researchers

This toolkit is designed for ease of use across domains. Whether you're in public health, social science, or retail analytics, `clustflow` offers interpretable clustering tools with simple integration into your workflow.


---

## 📦 Installation

```cmd
pip install -r requirements.txt
```

---

## 📘 Getting Started Example

```python
from clustflow.utils.seed import set_seed
from clustflow.preprocessing import Imputer, Encoder
from clustflow.dimensionality import PCAReducer
from clustflow.clustering import KMeansCluster
from clustflow.evaluation import compute_metrics
from clustflow.visualization import scatter_2d

# Set seed
set_seed(42)

# Impute missing values
imp = Imputer(strategy={'age': 'mean', 'income': 'median', 'cleanliness': ('group_median', 'zip3')})
df_clean = imp.fit_transform(df)

# Encode categorical columns
enc = Encoder(strategy='onehot')
X_encoded = enc.fit_transform(df_clean)

# Reduce dimensions
X_pca = PCAReducer(n_components=2).fit_transform(X_encoded)

# Cluster
kmeans = KMeansCluster(n_clusters=10)
labels = kmeans.fit_predict(X_pca)

# Evaluate
print(compute_metrics(X_pca, labels))

# Visualize
scatter_2d(X_pca, labels)
```

---

## 📊 Module Highlights

### `preprocessing.imputer.Imputer`

Flexible imputer for numeric columns. Accepts per-column strategies including `'mean'`, `'median'`, or `('group_median', 'zip3')`.

```python
Imputer(strategy={
    'age': 'mean',
    'score': ('group_median', 'region')
}).fit_transform(df)
```

---

### `preprocessing.encoder.Encoder`

Supports `'onehot'` and `'ordinal'` encoding. Can auto-detect categorical columns if none are passed.

```python
Encoder(strategy='onehot').fit_transform(df)
```

---

### `dimensionality.pca.PCAReducer`

Simple PCA reducer with labeled outputs.

```python
PCAReducer(n_components=2).fit_transform(X)
```

---

### `dimensionality.umap.UMAPReducer`

UMAP for non-linear dimensionality reduction. Random seed enabled.

```python
UMAPReducer(n_components=2).fit_transform(X)
```

---

### `clustering.kmeans.KMeansCluster`

Configurable KMeans wrapper with reproducibility.

```python
KMeansCluster(n_clusters=10, max_iter=500).fit_predict(X)
```

---

### `clustering.hierarchical.HierarchicalCluster`

Supports multiple linkage methods, distance thresholds, affinity metrics, and linkage matrix output for dendrograms.

```python
HierarchicalCluster(linkage_method='average', affinity='cosine').fit_predict(X)
```

---

### `evaluation.metrics.compute_metrics`

Returns silhouette, ARI, NMI, purity, Davies-Bouldin, etc.

```python
compute_metrics(X, labels)
```

---

### `evaluation.cluster_feature_importance.compute_feature_importance`

Finds features that separate clusters using ANOVA and Chi2.

```python
compute_feature_importance(X_df, labels, categorical_cols=['gender', 'region'])
```

---

### `evaluation.cluster_summary.summarize_clusters`

Summarizes feature distributions per cluster, including target label breakdowns if provided.

```python
summarize_clusters(X_df, labels, extra=df['target_label'])
```

---

### `visualization.scatter.scatter_2d`

2D scatter plot from PCA or UMAP, colored by cluster or label.

```python
scatter_2d(X_pca, labels, color_by=df['target_label'])
```

---

### `visualization.dendrogram.plot_dendrogram`

Plots a hierarchical dendrogram from linkage matrix or raw data.

---

### `visualization.feature_importance.plot_top_features`

Bar chart of top features based on ANOVA F or chi2 score.

---

### `visualization.cluster_summary.plot_cluster_heatmap`

Heatmap of per-cluster summary stats (mean, mode, ratios).

---

## 🚀 Coming Soon

- Deep clustering (DEC / IDEC / DCN)
- Shapley & attention-based feature explanations
- ZIP3-level geographic cluster summaries

---

## 💡 Tip for Domain Researchers

This toolkit is built to be used without deep ML expertise. Each module can be used standalone, and documentation includes walkthroughs and examples in the `examples/` folder.

---

## Usage

### FlexibleAttentionEmbedding

The `FlexibleAttentionEmbedding` class provides a flexible attention-based embedding layer for categorical and continuous features. It can optionally include a decoder for reconstruction. Example usage:

```python
from clustflow.embedding.attention_embedding import FlexibleAttentionEmbedding

# Example usage
cardinals = [10, 20, 30]  # Cardinalities of categorical features
emb_dims = [4, 8, 12]     # Embedding dimensions for categorical features
cont_dim = 5              # Number of continuous features
embed_dim = 64            # Output embedding dimension
n_heads = 4               # Number of attention heads
depth = 2                 # Number of transformer encoder layers
dropout = 0.1             # Dropout rate
output_dim = 100          # Output dimension for decoder

model = FlexibleAttentionEmbedding(cardinals, emb_dims, cont_dim, embed_dim, n_heads, depth, dropout, with_decoder=True, output_dim=output_dim)

# Forward pass
x_cat = torch.randint(0, 10, (32, len(cardinals)))  # Example categorical input
x_cont = torch.randn(32, cont_dim)                 # Example continuous input
embedding, reconstruction = model(x_cat, x_cont)
print(embedding.shape)  # Should be [32, embed_dim]
print(reconstruction.shape)  # Should be [32, output_dim]
```

### train_embedding

The `train_embedding` function allows training the `FlexibleAttentionEmbedding` model in an unsupervised manner using reconstruction loss. Example usage:

```python
from clustflow.embedding.train import train_embedding

# Example data
x_cat = torch.randint(0, 10, (100, len(cardinals)))  # Example categorical input
x_cont = torch.randn(100, cont_dim)                 # Example continuous input

# Initialize the model
model = FlexibleAttentionEmbedding(cardinals, emb_dims, cont_dim, embed_dim, n_heads, depth, dropout, with_decoder=True, output_dim=output_dim)

# Train the model
trained_model = train_embedding(model, learning_rate=0.001, epochs=10, batch_size=32, x_cat=x_cat, x_cont=x_cont)
```

