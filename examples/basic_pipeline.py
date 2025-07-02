import pandas as pd

# mock data loading
df = pd.DataFrame({
    'age': [25, 30, None, 22, 35],
    'income': [50000, 60000, 55000, None, 70000],
    'neighborhood_clean': ['A', 'B', 'A', 'C', None],
    'controlled': [1, 0, 1, 0, 1]  # Example target variable
})

# Handle missing data
from clustflow.dimensionality import PCAReducer
from clustflow.preprocessing import Imputer
from clustflow.clustering import KMeansCluster
imputer = Imputer(strategy={
    'age': 'mean',
    'income': 'median',
    'neighborhood_clean': ('group_median', 'zip3')
})
df_filled = imputer.fit_transform(df)

# Encode categoricals
from clustflow.preprocessing import Encoder
encoder = Encoder(strategy='onehot')
X = encoder.fit_transform(df_filled)

# Dimensionality reduction (Optional)
X_pca = PCAReducer(n_components=2).fit_transform(X)

# Clustering
model = KMeansCluster(n_clusters=8)
labels = model.fit_predict(X_pca)

# Evaluation
from clustflow.evaluation import compute_metrics
metrics = compute_metrics(X_pca, labels, true_labels=df.get('controlled'))
print(metrics)

# Visualization
from clustflow.visualization import scatter_2d
scatter_2d(X_pca, labels, color_by=df.get('controlled'))

# Feature importance (if needed)
from clustflow.evaluation import compute_feature_importance
from clustflow.visualization import plot_top_features

importance = compute_feature_importance(X, labels, categoricalcols=encoder.categorical_cols)
plot_top_features(importance['numerical'], title="Top Numerical Features")
plot_top_features(importance['categorical'], title="Top Categorical Features")

from clustflow.embedding import FlexibleAttentionEmbedding, train_embedding

# Embedding
cardinals = [10, 20, 30]  # Cardinalities of categorical features
emb_dims = [4, 8, 12]     # Embedding dimensions for categorical features
cont_dim = 5              # Number of continuous features
embed_dim = 64            # Output embedding dimension
n_heads = 4               # Number of attention heads
depth = 2                 # Number of transformer encoder layers
dropout = 0.1             # Dropout rate
output_dim = X.shape[1]   # Output dimension for decoder

embedding_model = FlexibleAttentionEmbedding(cardinals, emb_dims, cont_dim, embed_dim, n_heads, depth, dropout, with_decoder=True, output_dim=output_dim)
embedding_model = train_embedding(embedding_model, learning_rate=0.001, epochs=10, batch_size=32, x_cat=X[:, :len(cardinals)], x_cont=X[:, len(cardinals):])

# Use embeddings
embeddings, reconstruction = embedding_model(X[:, :len(cardinals)], X[:, len(cardinals):])