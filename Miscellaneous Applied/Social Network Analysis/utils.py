"""
Social Network Analysis — Utility Functions
Graph analysis of a real friendship network (SNAP Facebook ego-networks) with
networkx: degree, clustering, centrality, and community detection.
"""
import pandas as pd
import networkx as nx

def load_graph(path="data/facebook_edges.csv"):
    """Build an undirected graph from the edge list."""
    e = pd.read_csv(path)
    return nx.from_pandas_edgelist(e, "source", "target")

def basic_stats(G):
    """Core structural metrics of the graph."""
    degs = [d for _, d in G.degree()]
    return {
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "density": nx.density(G), "avg_degree": sum(degs)/len(degs),
        "max_degree": max(degs), "avg_clustering": nx.average_clustering(G),
        "components": nx.number_connected_components(G),
    }

def top_central(G, k_sample=400, top=10, seed=42):
    """Top nodes by (approx) betweenness and by degree centrality."""
    bc = nx.betweenness_centrality(G, k=k_sample, seed=seed)
    dc = nx.degree_centrality(G)
    top_bc = sorted(bc.items(), key=lambda x: -x[1])[:top]
    top_dc = sorted(dc.items(), key=lambda x: -x[1])[:top]
    return top_bc, top_dc, bc

def detect_communities(G):
    """Greedy-modularity community partition; returns list of node-sets."""
    return list(nx.community.greedy_modularity_communities(G))
