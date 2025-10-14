import networkx as nx
import random
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import metrics
import sklearn.datasets as ds
from louvain import louvain_algorithm
from girvan_newman import GirvanNewman, GirvanNewman_T
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


def get_percent_of_list(l, p):
    num_to_sample = int(len(l) * p)
    return random.sample(l, k=num_to_sample)

def create_k_means_graph(sample_size, cluster_num, shape, percent_of_original):
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
    labels = clustering.labels_
    cluster_points = [[] for i in range(cluster_num)]
    node_positions = {}
    g = nx.Graph()
    for i, (x, y) in enumerate(X):
        label = labels[i]
        g.add_node(i, pos=(x, y), cluster=label)
        node_positions[i] = (x, y)
        cluster_points[label].append(i)
    for label in range(int(cluster_num*percent_of_original)):
        nodes = cluster_points[label]
    #     nodes = get_percent_of_list(cluster_points[label], percent_of_original) + get_percent_of_list([
    # [sub_list for sub_list in inner_list if sub_list != label] for inner_list in cluster_points], (1-percent_of_original))
# + cluster_points[random.randint(0, cluster_num)] own points + other points
        print(nodes)
        for node in nodes:
            for _ in range(max(1, (cluster_num) // 2)):
                target = random.choice(nodes)
                if target != node:
                    x1, y1 = node_positions[node]
                    print("target", target)
                    x2, y2 = node_positions[target]
                    dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    g.add_edge(node, target, weight=dist)
    inter_edges_per_cluster = max(1, cluster_num // 2)
    for i in range(int(cluster_num*(1-percent_of_original))):
    #     nodes = get_percent_of_list(cluster_points[label], (1-percent_of_original)) + get_percent_of_list([
    # [sub_list for sub_list in inner_list if sub_list != label] for inner_list in cluster_points], (percent_of_original))
        for k in range(inter_edges_per_cluster):
            src = random.choice(cluster_points[i])#random.choice(cluster_points[i]) # pick a cluster to start an edge from
            # pick a different cluster
            other_clusters = [j for j in range(cluster_num) if j != i] 
            tgt_cluster = random.choice(other_clusters)
            tgt = random.choice(cluster_points[tgt_cluster])
            x1, y1 = node_positions[src]
            x2, y2 = node_positions[tgt]
            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            g.add_edge(src, tgt, weight=dist)
    colors = [labels[i] for i in range(len(X))]
    # nx.draw_networkx(
    #     g,
    #     pos=node_positions,
    #     node_color=colors,
    #     node_size=40,
    #     cmap=plt.cm.viridis,
    #     with_labels=False
    # )
    # plt.show()
    return g, cluster_points

def create_spectral_graph(sample_size, cluster_num, shape):
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
    clustering = SpectralClustering(n_clusters=cluster_num, gamma=10)
    clustering.fit(X)
    labels = clustering.labels_
    cluster_points = []
    node_positions = {}
    g = nx.Graph()
    g = nx.Graph()
    for i, (x, y) in enumerate(X):
        label = labels[i]
        g.add_node(i, pos=(x, y), cluster=label)
        node_positions[i] = (x, y)
        cluster_points[label].append(i)
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
    colors = [labels[i] for i in range(len(X))]
    # nx.draw_networkx(
    #     g,
    #     pos=node_positions,
    #     node_color=colors,
    #     node_size=40,
    #     cmap=plt.cm.viridis,
    #     with_labels=False
    # )
    #plt.show()
    return g, cluster_points

graphs = []
intervals = [i for i in np.arange(0.001, 0.05, 0.001)]
for i in range(len(intervals)):
    graphs.append(create_k_means_graph(30, 5, 3, 1)) #300 works for L, doesn't for G
graphs_for_storage_tests = []
for i in range(2, len(intervals), 1):
    graphs_for_storage_tests.append(create_k_means_graph(2*i, 4, 3, 1))
graphs_for_difficulty_tests = []
for i in np.arange(1, 0, -.01):
    graphs_for_difficulty_tests.append(create_k_means_graph(30, 5, 3, i))

def louvain_quality(intervals):
    louvain_quality = []
    louvain_communities = []
    louvain_execution_time = []
    louvain_memory = []
    louvain_difficulty = []
    for time in intervals:
        s, clusters_ = graphs[intervals.index(time)]
        c = louvain_algorithm(s)
        louvain_communities.append(c.get('communities'))
        louv_ = list(c.get('communities').items())
        louv_comm = list(c.get('communities').values())
        compare_true =  [None for i in range(len(s.nodes()))]#{"a": [0, 1, 2, 7], 1: [8, 9], 2:[3, 4, 5, 6]}
        
        compare_pred = [None for i in range(len(s.nodes()))]
        for l in clusters_:
            for i in l:
                compare_true[i] = clusters_.index(l)
        print(louv_)
        for key, value in louv_:
            for i in value:
                compare_pred[i] = key
        #print("compare true", compare_true)
        #print("compare pred", compare_pred)
        compare_score = metrics.rand_score(compare_true, compare_pred)
        #print(compare_score)#cluster_compare(clusters_, louv_comm)
        louvain_quality.append(compare_score)
        print("computing quality results for louvain...")


    for time in intervals:
        print("len intervals", len(intervals))
        s, clusters_ = graphs[intervals.index(time)]
        c = louvain_algorithm(s, time_limit=time)
        print(c)
        if "graph_at_time_limit" in c:
            louvain_execution_time.append(len(c.get("graph_at_time_limit").get("communities")))
        else:
            louvain_execution_time.append(len(c.get("communities")))
        print(f"computing t interval results and storage results for louvain... {time}")

    for i in graphs_for_storage_tests:
        s, clusters_ = graphs_for_storage_tests[graphs_for_storage_tests.index(i)]
        c = louvain_algorithm(s)
        #print(c)
        louvain_memory.append(c.get("memory_used_kb"))
        print("computing memory results for louvain...")
    
    for i in graphs_for_difficulty_tests:
        s, clusters_ = graphs_for_difficulty_tests[graphs_for_difficulty_tests.index(i)]
        c = louvain_algorithm(s)
        louvain_communities.append(c.get('communities'))
        louv_ = list(c.get('communities').items())
        compare_true =  [None for i in range(len(s.nodes()))]
        compare_pred = [None for i in range(len(s.nodes()))]
        for l in clusters_:
            for i in l:
                compare_true[i] = clusters_.index(l)
        print(louv_)
        for key, value in louv_:
            for i in value:
                compare_pred[i] = key
        #print("compare true", compare_true)
        #print("compare pred", compare_pred)
        compare_score = metrics.rand_score(compare_true, compare_pred)
        #print(compare_score)#cluster_compare(clusters_, louv_comm)
        louvain_difficulty.append(compare_score)
        print("computing difficulty results for louvain...")
    return louvain_quality, louvain_execution_time, louvain_memory, louvain_difficulty

def girvan_quality(intervals):
    girvan_quality = []
    girvan_communities = []
    girvan_execution_time = []
    girvan_memory = []
    girvan_difficulty = []
    for time in intervals:
        s, clusters_ = graphs[intervals.index(time)]
        c = GirvanNewman(s, len(clusters_))
        comm = c[0]
        girvan_communities.append(comm)
        girv_ = comm#list(c.get('communities').items())
        #print(comm)
        girv_comm = list(comm.values())
        for key, value_set in comm.items():
            girv_[key] = list(value_set)
        print(girv_)
        compare_true =  [None for i in range(len(s.nodes()))]#{"a": [0, 1, 2, 7], 1: [8, 9], 2:[3, 4, 5, 6]}
        compare_pred = [None for i in range(len(s.nodes()))]
        for l in clusters_:
            for i in l:
                compare_true[i] = clusters_.index(l)
        for key, value in girv_.items():
            for i in value:
                compare_pred[i] = key
        print("compare true", compare_true)
        print("compare pred", compare_pred)
        compare_score = metrics.rand_score(compare_true, compare_pred)
        girvan_quality.append(compare_score)
        print("computing quality results for Girvan-Newman...")

    for time in intervals:
        
        s, clusters_ = graphs[intervals.index(time)]
        c = GirvanNewman(s, len(clusters_))
        exec_time = len(c[0])
        print(c)
        girvan_execution_time.append(exec_time)
        print("computing t interval results for Girvan-Newman...")

    for i in graphs_for_storage_tests:
        s, clusters_ = graphs_for_storage_tests[graphs_for_storage_tests.index(i)]
        c2 = GirvanNewman_T(s, time)
        memory_used = c2[2]
        girvan_memory.append(memory_used)
        print("computing memory results for Girvan-Newman...")

    for i in graphs_for_difficulty_tests:
        s, clusters_ = graphs_for_difficulty_tests[graphs_for_difficulty_tests.index(i)]
        c = GirvanNewman(s, len(clusters_))
        comm = c[0]
        girvan_communities.append(comm)
        girv_ = comm#list(c.get('communities').items())
        #print(comm)
        girv_comm = list(comm.values())
        for key, value_set in comm.items():
            girv_[key] = list(value_set)
        print(girv_)
        compare_true =  [None for i in range(len(s.nodes()))]#{"a": [0, 1, 2, 7], 1: [8, 9], 2:[3, 4, 5, 6]}
        compare_pred = [None for i in range(len(s.nodes()))]
        for l in clusters_:
            for i in l:
                compare_true[i] = clusters_.index(l)
        for key, value in girv_.items():
            for i in value:
                compare_pred[i] = key
        print("compare true", compare_true)
        print("compare pred", compare_pred)
        compare_score = metrics.rand_score(compare_true, compare_pred)
        girvan_difficulty.append(compare_score)
        print("computing difficulty results for Girvan-Newman...")
    return girvan_quality, girvan_execution_time, girvan_memory, girvan_difficulty

def infomap_quality(intervals):
    infomap_quality = []
    infomap_communities = []
    infomap_execution_time = []
    infomap_memory = []
    infomap_difficulty = []
    for time in intervals:
        s, clusters_ = graphs[intervals.index(time)]
        map = Infomap(s, weight=False)
        c = map.run()
        comm = c[0]
        infomap_communities.append(comm)
        info_comm = list(comm.values())
        compare_true =  [None for i in range(len(s.nodes()))]#{"a": [0, 1, 2, 7], 1: [8, 9], 2:[3, 4, 5, 6]}
        info_ = comm
        compare_pred = [None for i in range(len(s.nodes()))]
        for l in clusters_:
            for i in l:
                compare_true[i] = clusters_.index(l)
        print(info_)
        for key, value in info_.items():
            for i in value:
                compare_pred[i] = key
        #print("compare true", compare_true)
        #print("compare pred", compare_pred)
        compare_score = metrics.rand_score(compare_true, compare_pred)
        infomap_quality.append(compare_score)
        print("computing quality results for Infomap...")
        #for time in intervals:

    for time in intervals:
        s, clusters_ = graphs[intervals.index(time)]
        map = Infomap(s, False)
        c = map.run()
        exec_time = c[1]
        print(c)
        infomap_execution_time.append(len(c[0]))
        print("computing t interval results for Infomap...")

    # for i in graphs_for_storage_tests:
    #     s, clusters_ = graphs_for_storage_tests[graphs_for_storage_tests.index(i)]
    #     map = Infomap(s, False)
    #     c = map.run()
    #     memory_used = c[2]
    #     infomap_memory.append(memory_used)
    #     print(f"computing memory results for Infomap... graph {graphs_for_storage_tests.index(i)}")
    # infomap_memory.pop(0)
    # infomap_memory.pop(0)

    for i in graphs_for_difficulty_tests:
        s, clusters_ = graphs_for_difficulty_tests[graphs_for_difficulty_tests.index(i)]
        map = Infomap(s, weight=False)
        c = map.run()
        comm = c[0]
        infomap_communities.append(comm)
        info_comm = list(comm.values())
        compare_true =  [None for i in range(len(s.nodes()))]#{"a": [0, 1, 2, 7], 1: [8, 9], 2:[3, 4, 5, 6]}
        info_ = comm
        compare_pred = [None for i in range(len(s.nodes()))]
        for l in clusters_:
            for i in l:
                compare_true[i] = clusters_.index(l)
        print(info_)
        for key, value in info_.items():
            for i in value:
                compare_pred[i] = key
        #print("compare true", compare_true)
        #print("compare pred", compare_pred)
        compare_score = metrics.rand_score(compare_true, compare_pred)
        infomap_difficulty.append(compare_score)
        print("computing difficulty results for Infomap...")
    return infomap_quality, infomap_execution_time, infomap_memory, infomap_difficulty

# def cluster_compare(actual, predicted):
#     correspondence = {}
#     for i, l in enumerate(predicted):
#         result = []
#         for other_list in actual:
#             intersection = len(set(l) & set(other_list))
#             union = len(set(l) | set(other_list))
#             res = intersection / float(union) if union != 0 else 0
#             result.append(res)
#         correspondence[i] = max(result)
#     result = []
#     for i, actual_cluster in enumerate(actual):
#         diff = []
#         for num in actual_cluster:
#             corr_value = correspondence.get(i, 0)
#             diff.append(num - corr_value)
#         result.append(sum(diff))
#     return sum(result)

Lquality, Ltime, Lmemory, Ldifficulty = louvain_quality(intervals)
Gquality, Gtime, Gmemory, Gdifficulty = girvan_quality(intervals)
Iquality, Itime, Imemory, Idifficulty = infomap_quality(intervals) # takes a while

#plt.subplot(1, 3, 1)
# plt.title("quality/time with K-means")
# plt.scatter(intervals, Lquality, label="Louvain")
# plt.scatter(intervals, Gquality, label="Girvan-Newman")
# plt.scatter(intervals, Iquality, label="Infomap")
# plt.xlabel("time (s)")
# plt.ylabel("accuracy score")
#plt.ylim(0, 1)
# plt.legend()

# plt.show()
#plt.subplot(1, 3, 1)
# plt.title("communities/time with 5 clusters")
# x_count = np.arange(0, t_intervals, 1).tolist()
# #print(x_count, Gtime)
# print(len(intervals), len(Gtime))
# plt.scatter(intervals, Ltime, label="Louvain")
# plt.scatter(intervals, Gtime, label="Girvan-Newman")
# plt.scatter(intervals, Itime, label="Infomap")
# plt.xlabel("time (s)")
# plt.ylabel("number of communities")
# plt.ylim(0, 15)
#plt.legend()

#plt.show()

#plt.subplot(1, 3, 2)
# plt.title("storage used/graph size with K-means")
# x_ = np.arange(0, len(graphs_for_storage_tests), 1).tolist()
# print(len(x_), len(Imemory))
# plt.scatter(x_, Lmemory, label="Louvain") 
# plt.scatter(x_, Gmemory, label="Girvan-Newman")
# print(intervals, Gmemory)
# plt.scatter(x_, Imemory, label="Infomap")
# plt.xlabel("graph size (nodes)")
# plt.ylabel("storage used (KB)")
# plt.legend()


#fix, (ax1, ax2) = plt.subplot(1, 1, 1)
plt.title("quality/graph difficulty")
x_ = np.arange(1, 0, -.01).tolist()
print(len(x_), len(Idifficulty))
plt.scatter(x_, Ldifficulty, label="Louvain")
plt.scatter(x_, Gdifficulty, label="Girvan-Newman")
plt.scatter(x_, Idifficulty, label="Infomap")
plt.xlabel("difficulty (p-r)")
plt.ylabel("quality score")
plt.xlim(1, 0)
#plt.ylim(0, 1)
plt.legend()
plt.show()

# Q/difficulty (p-r) change probability of connections between clusters and outside clusters
# C/t 