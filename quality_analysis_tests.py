import networkx as nx
import random
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
import sklearn.datasets as ds
from louvain import louvain_algorithm
import math
def create_random_graph(node_num, l, h):
    edge_num = int(node_num*random.randint(l, h))
    g = nx.Graph()
    nodes = random.sample(range(0, node_num), node_num)
    g.add_nodes_from(nodes)
    edges = []
    for i in range(edge_num):
        v_a = random.sample(nodes, 1)[0]
        v_b = random.sample(nodes, 1)[0]
        temp = tuple([v_a, v_b])
        if v_a != v_b and temp not in edges:
            edges.append(temp)
        else:
            i -= 1
    print(nodes, edges)  
    elements = set(element for tup in edges for element in tup)
    for node in nodes:
        if node not in elements:
            v_a = random.sample(nodes, 1)[0]
            print("nodes", nodes)
            print("node", node)
            e = nodes.pop(nodes.index(node))
            print("nodes", nodes)
            v_b = random.sample(nodes, 1)[0]
            nodes.append(e)
            temp = tuple([v_a, v_b])
            edges.append(temp)
    g.add_edges_from(edges)
    if nx.is_connected(g):
        nx.draw_networkx(g)
        return g
    else:
        create_random_graph(node_num, l, h)

def color_networks():
    pass

#create_random_graph(10, 1, 1)

def create_k_means_graph(sample_size, cluster_num, shape):
    X, _ = ds.make_circles(n_samples=sample_size, shuffle=False, noise=0.1)
    kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=10) 
    match shape:
        case 1:
            X, _ = ds.make_blobs(n_samples=sample_size, centers=cluster_num, cluster_std=0.60, random_state=0)
            title = "make_blobs"
        case 2:
            X, _ = ds.make_moons(n_samples=sample_size, noise=0.1)
            title = "make_moons"
        case 3:
            X, _ = ds.make_circles(n_samples=sample_size, factor=0.33, noise=0.1)
            title = "make_circles"
        case 4:
            X, _ = ds.make_friedman1(n_features=sample_size, noise=0.1)
            title = "make_friedman"
    clustering = SpectralClustering(n_clusters=cluster_num, random_state=0, n_init=10)
    kmeans.fit(X)
    clustering.fit(X)
    labels = clustering.labels_ #
    centers = kmeans.cluster_centers_
    cluster_points = []
    graphs = []
    for i in range(len(labels)):
        cluster_points.append(X[labels == i])
    for i in range(cluster_num):
        size = len(cluster_points[i][:, 0])
        g = nx.Graph()
        for j in range(size):
            g.add_node(j)
        # closest_to_center = find_point(cluster_points, i)
        # for j in range(random.randint(size, math.comb(size, 2))):
        #     g.add_edge()

        graphs.append(g)
        print(graphs[i])

    #edge_num = random.randint()
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title(title)
    plt.xlabel("ft1")
    plt.ylabel("ft2")
    plt.show()

def create_spectral_graph(sample_size, cluster_num, shape):
    #X, _ = make_blobs(n_samples=sample_size, centers=4, cluster_std=0.60, random_state=0)
    #X, _ = make_circles(n_samples=sample_size, shuffle=False, noise=0.1)
    match shape:
        case 1:
            X, _ = ds.make_blobs(n_samples=sample_size, centers=cluster_num, cluster_std=0.60, random_state=0)
            title = "make_blobs"
        case 2:
            X, _ = ds.make_moons(n_samples=sample_size, noise=0.1)
            title = "make_moons"
        case 3:
            X, _ = ds.make_circles(n_samples=sample_size, factor=0.33, noise=0.1)
            title = "make_circles"
        case 4:
            X, _ = ds.make_friedman1(n_features=sample_size, noise=0.1)
            title = "make_friedman"
    #clustering = SpectralClustering(n_clusters=cluster_num, random_state=0, n_init=10)
    clustering = SpectralClustering(n_clusters=cluster_num, gamma=10)

    clustering.fit(X)
    labels = clustering.labels_
    cluster_points = []
    graphs = []
    for i in range(len(labels)):
        cluster_points.append(X[labels == i])
    for i in range(cluster_num):
        size = len(cluster_points[i][:, 0])
        g = nx.Graph()
        for j in range(size):
            g.add_node(j)
        # closest_to_center = find_point(cluster_points, i)
        # for j in range(random.randint(size, math.comb(size, 2))):
        #     g.add_edge()

        graphs.append(g)
        print(graphs[i])
    # for i in range(len(labels)):
    #     plt.scatter(cluster_points[i][:, 0], cluster_points[i][:, 1])
    for i in range(len(labels)):
        plt.scatter(cluster_points[i][:, 0], cluster_points[i][:, 1])
    #plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    #plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title(title)
    plt.xlabel("ft1")
    plt.ylabel("ft2")
    plt.show()

def find_point(c, i):
    s = sum(c[i][:, 0]) 
    print(c[i][:, 0])
    print(s)
    return
create_k_means_graph(1000, 5, 1)
create_k_means_graph(1000, 5, 2)
create_k_means_graph(1000, 5, 3)
create_k_means_graph(1000, 5, 4)
create_spectral_graph(1000, 5, 1)
create_spectral_graph(1000, 5, 2)
create_spectral_graph(1000, 5, 3)
create_spectral_graph(1000, 5, 4)
#a = create_random_graph(10, 1, 1)
#a = create_random_graph(5, 1, 1)
#nx.draw_networkx(a)

#c, s = louvain_algorithm(a)

# input_sparse_graphs = []
# input_dense_graphs = []
# for i in range(500):
#     node_num = random.randint(0, 1000)
#     g = create_random_graph(random.randint(0, 1000), 1, 1)#node_num%2)
#     input_sparse_graphs.append(g)
# for i in range(500):
#     node_num = random.randint(0, 1000)
#     g = create_random_graph(random.randint(0, 1000), 3, 5)
#     input_dense_graphs.append(g)
# output_sparse_alg_1 = []
# output_dense_alg_1 = []
# for i in range(500):
#     output_sparse_alg_1.append(louvain_algorithm(input_sparse_graphs[i]))
#     output_dense_alg_1.append(louvain_algorithm(input_dense_graphs[i]))
# output_sparse_alg_2 = []
# output_dense_alg_2 = []
# for i in range(500):
#     output_sparse_alg_2.append(alg_2(input_sparse_graphs[i]))
#     output_dense_alg_2.append(alg_2(input_dense_graphs[i]))
# output_sparse_alg_3 = []
# output_dense_alg_3 = []
# for i in range(500):
#     output_sparse_alg_3.append(alg_3(input_sparse_graphs[i]))
#     output_dense_alg_3.append(alg_3(input_dense_graphs[i]))

# plt.plot(input_sparse_graphs, output_sparse_alg_1, label="louvain")
# plt.plot(input_sparse_graphs, output_sparse_alg_2, label="alg2")
# plt.plot(input_sparse_graphs, output_sparse_alg_3, label="alg3")

