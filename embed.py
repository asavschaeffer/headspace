import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import hdbscan

# Use a pre-trained model. Using CUDA for GPU acceleration.
model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1', device='cuda')

def get_embedding(file_path):
    """Reads a file and returns its embedding."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return model.encode(content, convert_to_numpy=True)

if len(sys.argv) == 2:
    # --- Single File Embedding ---
    # This path is called by the Go program for each file individually.
    file_path = sys.argv[1]
    embedding = get_embedding(file_path).tolist()
    print(json.dumps(embedding))

elif len(sys.argv) == 4 and sys.argv[3] == '--cluster':
    # --- Clustering and Visualization ---
    # This path is called once by the Go program with all the embeddings.
    embedding_file = sys.argv[1]
    output_file = sys.argv[2]
    
    with open(embedding_file, 'r') as f:
        embeddings_data = json.load(f)

    # Extract file paths and embeddings
    files = [item['path'] for item in embeddings_data]
    embeddings = np.array([item['embedding'] for item in embeddings_data])

    # --- HDBSCAN Clustering ---
    # Use HDBSCAN to find natural clusters. min_cluster_size is a key parameter to tune.
    # It won't force every point into a cluster; outliers are labeled -1.
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True)
    clusters = clusterer.fit_predict(embeddings)
    
    # --- Dimensionality Reduction for Visualization ---
    # Reduce the 512-dimensional embeddings to 2D for plotting.
    # Perplexity is typically 5-50. It should be less than the number of samples.
    perplexity = min(30, len(embeddings) - 1)
    if perplexity > 0:
        tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
    else:
        # Handle case with very few documents
        embeddings_2d = np.zeros((len(embeddings), 2))

    # --- Create DataFrame for Plotting ---
    df = pd.DataFrame({
        'file': [f.split('\\')[-1] for f in files], # Show clean filename
        'path': files,
        'cluster': [f'Cluster {c}' if c != -1 else 'Outlier' for c in clusters],
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1]
    })

    # --- Generate Interactive Plotly Chart ---
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='cluster',
        hover_name='file',
        hover_data={'path': True, 'cluster': True, 'x': False, 'y': False},
        title='Document Cluster Map'
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
    fig.write_html("output/clusters.html")

    # --- Save Cluster Data for Go Program ---
    # The Go program will use this JSON to understand the clusters.
    cluster_data = []
    for cluster_id in np.unique(clusters):
        cluster_files = [files[i] for i, c in enumerate(clusters) if c == cluster_id]
        cluster_data.append({"cluster_id": int(cluster_id), "files": cluster_files})

    output_data = {"embeddings": embeddings_data, "clusters": cluster_data}
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Successfully created cluster map: output/clusters.html")
