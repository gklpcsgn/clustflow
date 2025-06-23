from clustflow.utils.seed import set_seed
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")  # or whatever

# Set reproducibility
set_seed(42)

# Handle missing data
from clustflow.preprocessing import Imputer
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
from clustflow.dimensionality import PCAReducer
X_pca = PCAReducer(n_components=2).fit_transform(X)

# Clustering
from clustflow.clustering import KMeansCluster
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