from scipy.sparse import csr_matrix
import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx

def mean_distance_in_cluster(X_spatial, cluster_labels, cluster):
    """
    Calculate the mean distance among nodes within the same cluster.
    """
    cluster_distances = []
    n = X_spatial.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            if cluster_labels[i] == cluster and cluster_labels[j] == cluster:
                distance = np.linalg.norm(X_spatial[i] - X_spatial[j])  # Calculate Euclidean distance
                cluster_distances.append(distance)

    if cluster_distances:
        mean_distance = np.mean(cluster_distances)
    else:
        mean_distance = 0.0  # Default value when no nodes are in the same cluster

    return mean_distance

def cal_weighted_labeled_adj(X_spatial, cluster_labels, cluster):
    '''
    Calculate weighted adjacency matrix based on KNN
    For each row of X, put an edge between nodes i and j
    If nodes are among the n_neighbors nearest neighbors of each other
    according to Euclidean distance, and their cluster labels are both '1'
    '''
    n = X_spatial.shape[0]
    #set distance threshold to 2/3 of mean distance in cluster
    distance_threshold = mean_distance_in_cluster(X_spatial, labels, cluster)/1.75
    # Compute pairwise spatial distances
    dist = Eu_dis(X_spatial)
    
    # Calculate the threshold based on the mean squared distance
    t = np.mean(dist**2)
    gk_dist = gaussian_kernel(dist, t)
    #print('GKs:', gk_dist)
    #print(gk_dist)
    W_L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if cluster_labels[i] == cluster and cluster_labels[j] == cluster and dist[i, j] <= distance_threshold:
                W_L[i, j] = gk_dist[i, j]
                W_L[j, i] = gk_dist[i, j]

    return W_L

def dge_laplace_score_g(X_gene, gene, G):
    G_sparse = csr_matrix(G)
    degrees = np.array(G_sparse.sum(axis=1)).flatten()
    D = np.diag(degrees)

    # Calculate mu_i (mean)
    mu_i = np.sum(D.diagonal() * X_gene[:, gene]) / np.sum(D.diagonal())
    # Calculate var_g (variance)
    var_g = np.sum(D.diagonal() * (X_gene[:, gene] - mu_i)**2)

    # Calculate the Laplacian score
    laplacian_score_i = 0
    for u, v in zip(*G_sparse.nonzero()):
        laplacian_score_i += (G_sparse[u, v] * (X_gene[u, gene] - X_gene[v, gene])**2) / var_g

    name = adata.var_names[gene]
    return laplacian_score_i, gene, name

def dge_laplace_scores(X_spatial, X_gene, labels, cluster):
    G = cal_weighted_labeled_adj(X_spatial, labels, cluster)
    print(f'Graph Structure of Cluster {cluster} Constructed')
    X_gene = X_gene.toarray()
    laplace_scores = []
    print('Calculating Laplace Scores')
    for i in range(X_gene.shape[1]):
        laplace_score_i, gene, name = dge_laplace_score_g(X_gene, i, G)
        laplace_scores.append((laplace_score_i, gene, name))
        if i % (X_gene.shape[1] // 10) == 0:
            print("-" * 2 *(i // (X_gene.shape[1] // 10)))
    return laplace_scores

sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=250)
adata = adata[:, adata.var['highly_variable']]
gene_names = adata.var_names
X_gene = adata.X
ls = dge_laplace_scores(X_spatial, X_gene, labels, '5')
print(ls)

import math
filtered_data = [tup for tup in ls if not math.isnan(tup[0])]

# Sort the filtered data
sorted_data = sorted(filtered_data, key=lambda x: x[0])

print(sorted_data)

cluster_label = '5'
graph_matrix = cal_weighted_labeled_adj(X_spatial, labels, cluster_label)
# Create a graph from the adjacency matrix
G = nx.Graph(graph_matrix)

# Extract spatial coordinates for plotting
spatial_coords = adata.obsm['spatial']

# Add all nodes to the graph, not just those connected by edges
all_nodes = range(len(spatial_coords))
G.add_nodes_from(all_nodes)

# Plot spatial coordinates as nodes
pos = {i: spatial_coords[i] for i in range(len(spatial_coords))}

# Reflect the figure vertically
for node in pos:
    pos[node] = (pos[node][0], -pos[node][1])

nx.draw(G, pos, with_labels=False, node_size=10)

# Plot edges connecting nodes
nx.draw_networkx_edges(G, pos, alpha=0.5)

# Customize plot appearance if needed
plt.title(f"Graph Structure for Cluster {cluster_label} ")
plt.axis('off')  # Turn off axis labels
save = f"C:/Users/cottre61/Downloads/labeled_graph{cluster_label}"
plt.savefig(save)
plt.show()

for gene_of_interest in ['SST', 'PRG4', 'UCHL1', 'SFTPB']:
    plt.rcParams["figure.figsize"] = (8, 8)
    sc.pl.spatial(adata, color=gene_of_interest,cmap = 'hot', save=f'gene_plot_{gene_of_interest}.png', size = 1.7) 
