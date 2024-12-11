import sys
import torch
import numpy as np
import networkx as nx
import json
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay
from PIL import Image
import torchvision.transforms as transforms
from safetensors.torch import load_file
from model import UNet
import argparse 
import os

def filter_long_edges(vertices, edges, max_length=50):
    filtered_edges = []
    for edge in edges:
        idx1, idx2 = edge
        v1, v2 = vertices[idx1], vertices[idx2]
        length = np.linalg.norm(v1 - v2)
        if length <= max_length:
            filtered_edges.append(edge)
    return filtered_edges

def build_graph_with_triangulation(edge_points, n_clusters=100, max_length=50):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    if n_clusters < 1:
        raise ValueError(f"Number of clusters (n_clusters) must be at least 1, got {n_clusters}.")
    
    if len(edge_points) < n_clusters:
        raise ValueError(f"Number of edge points ({len(edge_points)}) must be greater than or equal to the number of clusters ({n_clusters}).")
    
        
    kmeans.fit(edge_points)
    vertices = kmeans.cluster_centers_.astype(float)

    min_x, min_y = vertices.min(axis=0)
    max_x, max_y = vertices.max(axis=0)
    vertices[:, 0] = (vertices[:, 0] - min_x) / (max_x - min_x) 
    vertices[:, 1] = (vertices[:, 1] - min_y) / (max_y - min_y)

    tri = Delaunay(vertices)
    edges = set()
    for simplex in tri.simplices:
        edges.update([
            (simplex[0], simplex[1]),
            (simplex[0], simplex[2]),
            (simplex[1], simplex[2])
        ])

    filtered_edges = filter_long_edges(vertices, edges, max_length)

    G = nx.Graph()
    for idx, vertex in enumerate(vertices):
        G.add_node(idx, pos=(vertex[1], vertex[0]))  
    G.add_edges_from(filtered_edges)

    return G, vertices

def main(image_path, n_clusters=50, max_length=50):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' does not exist.")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    model = UNet()
    state_dict = load_file('result/checkpoint-3750/model.safetensors')
    model.load_state_dict(state_dict)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    image = Image.open(image_path).convert('RGB')
    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_image)['logits']
        logits = outputs.squeeze(0).cpu().numpy()
        edge_mask = (logits > 0.5).astype('uint8') * 255

    edge_points = np.column_stack(np.nonzero(edge_mask))
    G, vertices = build_graph_with_triangulation(edge_points, n_clusters, max_length)

    nodes = [{'id': int(idx), 'pos': [coord for coord in vertex]} for idx, vertex in enumerate(vertices)]
    edges = [{'source': int(edge[0]), 'target': int(edge[1])} for edge in G.edges()]

    graph_data = {'nodes': nodes, 'edges': edges}
    print(json.dumps(graph_data)) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Graph Extractor')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('n_clusters', type=int, help='Number of clusters (vertices)')
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        main(args.image_path, args.n_clusters)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)