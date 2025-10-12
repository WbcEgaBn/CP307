import networkx as nx
import random
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import metrics
import sklearn.datasets as ds
from louvain import louvain_algorithm
from girvan_newman import GirvanNewman
from infomap import Infomap
import numpy as np


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
        #nx.draw_networkx(g)
        return g
    else:
        create_random_graph(node_num, l, h)

def color_networks():
    pass

#create_random_graph(10, 1, 1)

def create_k_means_graph(sample_size, cluster_num, shape):
    # Generate data
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

    # Run clustering
    clustering = SpectralClustering(n_clusters=cluster_num, random_state=0, n_init=10)
    kmeans.fit(X)
    clustering.fit(X)

    labels = clustering.labels_
    centers = kmeans.cluster_centers_

    # --- create cluster → list of node indices ---
    cluster_points = [[] for _ in range(cluster_num)]
    node_positions = {}
    g = nx.Graph()

    # Add nodes and group by cluster
    for i, (x, y) in enumerate(X):
        label = labels[i]
        g.add_node(i, pos=(x, y), cluster=label)
        node_positions[i] = (x, y)
        cluster_points[label].append(i)

    # Add random edges within each cluster
    for label in range(cluster_num):
        nodes = cluster_points[label]
        for node in nodes:
            for _ in range(max(1, cluster_num // 2)):
                target = random.choice(nodes)
                if target != node:
                    x1, y1 = node_positions[node]
                    x2, y2 = node_positions[target]
                    dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    g.add_edge(node, target, weight=dist)

    # Add a few inter-cluster edges
    inter_edges_per_cluster = max(1, cluster_num // 2)
    for i in range(cluster_num):
        for _ in range(inter_edges_per_cluster):
            src = random.choice(cluster_points[i])
            # pick a different cluster
            other_clusters = [j for j in range(cluster_num) if j != i]
            tgt_cluster = random.choice(other_clusters)
            tgt = random.choice(cluster_points[tgt_cluster])
            x1, y1 = node_positions[src]
            x2, y2 = node_positions[tgt]
            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            g.add_edge(src, tgt, weight=dist)

    #Draw the graph with cluster-based colors
    colors = [labels[i] for i in range(len(X))]
    # nx.draw_networkx(
    #     g,
    #     pos=node_positions,
    #     node_color=colors,
    #     node_size=40,
    #     cmap=plt.cm.viridis,
    #     with_labels=False
    # )
    #plt.title(title)
    #plt.xlabel("ft1")
    #plt.ylabel("ft2")
    #plt.show()

    return g, cluster_points


# def create_k_means_graph(sample_size, cluster_num, shape):
#     #X, _ = ds.make_circles(n_samples=sample_size, shuffle=False, noise=0.1)
#     kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=10) 
#     match shape:
#         case 1:
#             X, _ = ds.make_blobs(n_samples=sample_size, centers=cluster_num, cluster_std=0.60, random_state=0)
#             title = "make_blobs"
#         case 2:
#             X, _ = ds.make_moons(n_samples=sample_size, noise=0.1)
#             title = "make_moons"
#         case 3:
#             X, _ = ds.make_circles(n_samples=sample_size, factor=0.33, noise=0.1)
#             title = "make_circles"
#         case 4:
#             X, _ = ds.make_friedman1(n_features=sample_size, noise=0.1)
#             title = "make_friedman"
#     clustering = SpectralClustering(n_clusters=cluster_num, random_state=0, n_init=10)
#     kmeans.fit(X)
#     clustering.fit(X)
#     labels = clustering.labels_ #
#     centers = kmeans.cluster_centers_
#     cluster_points = []
#     node_positions = {None for i in range(cluster_num)}
#     clusters_to_nodes = []
#     graphs = []

#     for i in range(cluster_num):
#         print(len(labels))
#         cluster_points.append(X[labels == i])
#         #clusters_to_nodes.append([ for k in range(len(labels))])
#     for i in range(cluster_num):

#         #print("cluster:", cluster_num)
#         size = len(cluster_points[i][:, 0])
#         g = nx.Graph()
#         graphs.append(g)
#         #node_positions[i] = (X)
#         # clusters_to_nodes.append([i for j in range(size)])
#         #node_positions[i] = (cluster_points[i][:, 0], cluster_points[i][:, 1])
#         #print(node_positions[i])
#         for j in range(size):
#             node_positions[i] = (cluster_points[i][:, 0][j], cluster_points[i][:, 1][j])
#             #print(cluster_points[i])
#             g.add_node(j)
            
#             for k in range(cluster_num // 2):
                
#                 #print(f"j={j}")
#                 node = random.choice(list(g.nodes()))
#                 #print(f"n:{node}")
#                 if j != node:
#                     #print(node_positions.get(j))
#                     jx = node_positions.get(j)[0]
#                     jy = node_positions.get(j)[1]
#                     nodex, nodey = node_positions.get(node)
#                     g.add_edge(j, node, weight=((nodex-jx) ** 2 + (nodey-jy) ** 2) ** .5)

#             #nx.draw_networkx(g)
#     g = nx.compose_all(graphs)
#     nx.draw_networkx(g)
#         #print(graphs[i])
#     return g, cluster_points
#     #edge_num = random.randint()
    
#     #plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
#     #plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
#     plt.title(title)
#     plt.xlabel("ft1")
#     plt.ylabel("ft2")
#     plt.show()

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
    # plt.title(title)
    # plt.xlabel("ft1")
    # plt.ylabel("ft2")
    # plt.show()

#a, X = create_k_means_graph(100, 5, 1)
# create_k_means_graph(1000, 5, 2)
# create_k_means_graph(1000, 5, 3)
# create_k_means_graph(1000, 5, 4)
# create_spectral_graph(1000, 5, 1)
# create_spectral_graph(1000, 5, 2)
# create_spectral_graph(1000, 5, 3)
# create_spectral_graph(1000, 5, 4)
#a = create_random_graph(5, 1, 1)
#print(f"l:{c}")
graphs = []
intervals = [i for i in np.arange(0.001, 0.05, 0.0005)]
for i in range(len(intervals)):
    graphs.append(create_k_means_graph(30, 5, 3)) #300 works for L, doesn't for G


def louvain_quality(intervals, t, t_intervals):
    louvain_quality = []
    louvain_communities = []
    louvain_execution_time = []
    louvain_memory = []
    
    for time in intervals:
        
        s, clusters_ = graphs[intervals.index(time)]
        #s = nx.random_regular_graph(20, 50)
        c = louvain_algorithm(s)
        louvain_communities.append(c.get('communities'))
        louv_comm = list(c.get('communities').values())
        #print("cluster length:", len(clusters_))
        #print("clusdters_0", clusters_[0])
        #print("actual", clusters_, len(clusters_))
        #print("c_l:", louv_comm_len, "c_n:", clust_num)
        #len_acc_coeff = (clust_num - louv_comm_len) / clust_num
        #print("predicted", louv_comm, len(louv_comm))
        compare = cluster_compare(clusters_, louv_comm)
        louvain_quality.append(compare)
        print("computing quality results for louvain...")
    '''
    'execution_time': 0.20398998260498047, 'memory_used_kb': 13.6484375, 'memory_used_bytes': 13976, 
    'stats': {'modularity_calculations': 6199, 'node_moves': 270, 'phases': 2, 'iterations': 6}}
    '''
    for t in range(t_intervals):
        
        s, clusters_ = graphs[intervals.index(time)]
        c = louvain_algorithm(s, time_limit=t)
        louvain_execution_time.append(c.get("execution_time"))
        print(c)
        print("computing t interval results and storage results for louvain...")
        louvain_memory.append(c.get("memory_used_kb"))# with len(g.nodes) on X
    # louvain_memory.append(memory)

    return louvain_quality, louvain_execution_time, louvain_memory

def girvan_quality(intervals, t, t_intervals):
    girvan_quality = []
    girvan_communities = []
    girvan_execution_time = []
    girvan_memory = []

    for time in intervals:
        
        s, clusters_ = graphs[intervals.index(time)]
        #s = nx.random_regular_graph(20, 50)
        c = GirvanNewman(s)
        comm = c[0]
        exec_time = c[1]
        memory_used = c[2]
        '''
        c = louvain_algorithm(s)
        louvain_communities.append(c.get('communities'))
        louv_comm = list(c.get('communities').values())
        '''
        girvan_communities.append(comm)
        girv_comm = list(comm.values())
        #print("cluster length:", len(clusters_))
        #print("clusdters_0", clusters_[0])
        #print("actual", clusters_, len(clusters_))
        #print("c_l:", louv_comm_len, "c_n:", clust_num)
        #len_acc_coeff = (clust_num - louv_comm_len) / clust_num
        #print("predicted", louv_comm, len(louv_comm))
        compare = cluster_compare(clusters_, girv_comm)
        girvan_quality.append(compare)
        print("computing quality results for Girvan-Newman...")
    '''
    'execution_time': 0.20398998260498047, 'memory_used_kb': 13.6484375, 'memory_used_bytes': 13976, 
    'stats': {'modularity_calculations': 6199, 'node_moves': 270, 'phases': 2, 'iterations': 6}}
    '''
    for t in range(t_intervals):
        
        s, clusters_ = graphs[intervals.index(time)]
        c = GirvanNewman(s)
        girvan_execution_time.append(None)
        print(c)
        print("computing t interval results and storage results for Girvan-Newman...")
        #c = louvain_algorithm(s)
        girvan_memory.append(c[2])# with len(g.nodes) on X
    # louvain_memory.append(memory)

    return girvan_quality, girvan_execution_time, girvan_memory

def infomap_quality():
    pass
    infomap_quality = []
    infomap_communities = []
    infomap_execution_time = []
    infomap_memory = []

    #for time in intervals:

# def cluster_compare(actual, predicted):
#     correspondence = {}
#     for l in predicted:
#     #     min_mag = 100000000000000
#         result = []
#         for other_list in actual:
            
#             res = len(set(l) and set(other_list)) / float(len(set(l) or set(other_list)))
#             result.append(res)
#             #print(res)
#         #print(result)
#         correspondence[predicted.index(l)] = min(result)
#         print(correspondence)

#     result = []
#     for list in actual:
#         diff = [None for i in range(len(actual))]
#         for num in list:
#             print("num", num)
#             print("list", list)
#             if predicted.index(l) is None:
#                 diff.append(0)
#             else:
#                 diff.append(num - correspondence.get(predicted.index(l))) #actual-predicted
#         if len(diff) is None:
#             diff = [0 for i in range (len(actual))]
#             print(diff)
#         result.append(sum(diff))

#     print(result)





def cluster_compare(actual, predicted):
    correspondence = {}

    # Compare each predicted cluster to each actual cluster
    for i, l in enumerate(predicted):
        result = []
        for other_list in actual:
            # Use proper set operations (intersection & union)
            intersection = len(set(l) & set(other_list))
            union = len(set(l) | set(other_list))
            res = intersection / float(union) if union != 0 else 0
            result.append(res)

        # Store the *max* similarity (best match), not min
        correspondence[i] = max(result)
        #print(f"Cluster {i} best similarity: {correspondence[i]:.2f}")

    # Compute some numeric comparison (this part seems conceptual)
    result = []
    for i, actual_cluster in enumerate(actual):
        diff = []
        for num in actual_cluster:
            # Find the corresponding predicted cluster’s score
            corr_value = correspondence.get(i, 0)
            diff.append(num - corr_value)
        result.append(sum(diff))

    # print("Correspondence:", correspondence)
    # print("Result:", result)
    return sum(result)

    # result = []
    # for value in correspondence.values():
    #print("ts", sum(correspondence.values())/len(correspondence.items()))
    #return sum(correspondence.values())/len(correspondence.items())
    


Lquality, Ltime, Lmemory = louvain_quality(intervals, 5, 7)
Gquality, Gtime, Gmemory = girvan_quality(intervals, 5, 7)
#imap = infomap_quality(intervals, 5, 7)

#print(len(intervals), len(quality))
plt.title("quality/time")
plt.scatter(intervals, Lquality, label="l")
plt.scatter(intervals, Gquality, label="g")
plt.xlabel("time (s)")
plt.ylabel("quality percent error")
plt.legend()
plt.show()