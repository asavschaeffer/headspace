
import os
import json
import argparse
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import hdbscan
import python-frontmatter
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_documents(input_dir):
    documents = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    post = python-frontmatter.load(f)
                    documents.append({
                        "Source": file_path,
                        "Title": post.metadata.get('title', 'No Title'),
                        "Content": post.content,
                        "Tokens": len(post.content.split())
                    })
    return documents

def generate_embeddings(documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    content = [doc['Content'] for doc in documents]
    embeddings = model.encode(content, show_progress_bar=True)
    for i, doc in enumerate(documents):
        doc['Embedding'] = embeddings[i].tolist()
    return documents

def cluster_embeddings(documents):
    embeddings = np.array([doc['Embedding'] for doc in documents])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    clusterer.fit(embeddings)
    for i, doc in enumerate(documents):
        doc['Cluster'] = clusterer.labels_[i]
    return documents, clusterer

def extract_keywords(documents):
    df = pd.DataFrame(documents)
    keywords = {}
    for cluster_id in df['Cluster'].unique():
        if cluster_id == -1:
            continue
        cluster_docs = df[df['Cluster'] == cluster_id]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
        vectorizer.fit(cluster_docs['Content'])
        keywords[str(cluster_id)] = vectorizer.get_feature_names_out().tolist()
    return keywords

def visualize_clusters(documents, clusterer):
    embeddings = np.array([doc['Embedding'] for doc in documents])
    df = pd.DataFrame(embeddings)
    df['Cluster'] = clusterer.labels_
    df['Title'] = [doc['Title'] for doc in documents]
    
    fig = px.scatter(df, x=0, y=1, color='Cluster', hover_data=['Title'])
    return fig

def main():
    parser = argparse.ArgumentParser(description="Analyze a directory of Markdown documents.")
    parser.add_argument("input_dir", help="Path to the directory containing Markdown files.")
    args = parser.parse_args()

    output_dir = os.path.join("output", f"run_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(output_dir, exist_ok=True)

    print("Parsing documents...")
    documents = parse_documents(args.input_dir)
    
    print("Generating embeddings...")
    documents = generate_embeddings(documents)
    
    print("Clustering documents...")
    documents, clusterer = cluster_embeddings(documents)
    
    print("Extracting keywords...")
    keywords = extract_keywords(documents)
    
    print("Visualizing clusters...")
    fig = visualize_clusters(documents, clusterer)

    with open(os.path.join(output_dir, "clustered_data.json"), 'w') as f:
        json.dump(documents, f, indent=4)
        
    with open(os.path.join(output_dir, "cluster_keywords.json"), 'w') as f:
        json.dump(keywords, f, indent=4)
        
    fig.write_html(os.path.join(output_dir, "cluster_visualization.html"))
    
    print(f"Analysis complete. Results are in {output_dir}")

if __name__ == "__main__":
    main()
