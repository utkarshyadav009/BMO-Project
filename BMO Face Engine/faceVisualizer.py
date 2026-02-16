# import pandas as pd
# import plotly.express as px

# df = pd.read_csv("face_tags.csv")
# print(type(df))
# print(df.head())

# # Only numeric columns
# num_df = df[['valence','arousal','control','novelty','obstruct']].copy()

# fig = px.parallel_coordinates(
#     num_df,
#     dimensions=num_df.columns,
#     color='valence'  # or 'arousal'
# )
# fig.show()



import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
import umap

# Load your CSV exported from C++
df = pd.read_csv("face_tags.csv")

# Extract 5D vectors
X = df[['valence','arousal','control','novelty','obstruct']].values

# -------------------
# Option 1: t-SNE
tsne = TSNE(n_components=3, perplexity=5, random_state=42)
X_tsne = tsne.fit_transform(X)

fig = px.scatter_3d(
    x=X_tsne[:,0], y=X_tsne[:,1], z=X_tsne[:,2],
    text=df['name'],
    color=df['valence'],        # optional: color by valence/arousal/etc
    size_max=5,
    opacity=0.8
)

# -------------------
# Option 2: UMAP (optional, preserves global structure better)
# reducer = umap.UMAP(n_components=3, random_state=42)
# X_umap = reducer.fit_transform(X)
# # Use X_umap instead of X_tsne in plotting

# # Create 3D scatter
# fig = px.scatter_3d(
#     x=X_umap[:,0], y=X_umap[:,1], z=X_umap[:,2],
#     text=df['name'],
#     color=df['valence'],        # optional: color by valence/arousal/etc
#     size_max=5,
#     opacity=0.8
# )
fig.update_traces(marker=dict(size=5))
fig.show()
