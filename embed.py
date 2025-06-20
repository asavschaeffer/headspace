import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1', device='cuda')

if len(sys.argv) == 2:
    # Embed single file
    file_path = sys.argv[1]
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    embedding = model.encode(content, convert_to_numpy=True).tolist()
    print(json.dumps(embedding))

elif len(sys.argv) == 4 and sys.argv[3] == '--cluster':
    # Cluster embeddings
    embedding_file = sys.argv[1]
    output_file = sys.argv[2]
    with open(embedding_file, 'r') as f:
        embeddings_data = json.load(f)

    embeddings = np.array([item['embedding'] for item in embeddings_data])
    files = [item['path'] for item in embeddings_data]

    num_clusters = min(5, len(embeddings))
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmeans.fit_predict(embeddings)

    cluster_data = []
    for cluster_id in range(num_clusters):
        cluster_files = [files[i] for i, c in enumerate(clusters) if c == cluster_id]
        cluster_data.append({"cluster_id": cluster_id, "files": cluster_files})

    output = {"embeddings": embeddings_data, "clusters": cluster_data}
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)