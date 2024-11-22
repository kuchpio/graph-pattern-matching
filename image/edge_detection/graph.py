import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay
from PIL import Image
import torchvision.transforms as transforms
from safetensors.torch import load_file
from model import UNet

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
    kmeans.fit(edge_points)
    vertices = kmeans.cluster_centers_.astype(int)

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

def visualize_graph(G, vertices):
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_size=50)
    plt.gca().invert_yaxis()
    plt.show()

def main(edge_img, n_clusters=50, max_length=50):
    edge_points = np.column_stack(np.nonzero(edge_img))
    G, vertices = build_graph_with_triangulation(edge_points, n_clusters, max_length)
    visualize_graph(G, vertices)


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

image_path = '../beaglee.jpg'
image = Image.open(image_path).convert('RGB')
input_image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_image)['logits']
    print(outputs)
    logits = outputs.squeeze(0).cpu().numpy()
    edge_mask = (logits > 0.5).astype('uint8') * 255
    
main(edge_mask)

