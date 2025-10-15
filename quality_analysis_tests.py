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
from sys import argv

# enter the parameters for the graph you want to see
# {plot_type}, {cluster_method}, {shapes} {dense_or_sparse (only if plot type is q/t)}
# more details are in the readme

def create_k_means_graph(sample_size, cluster_num, shape, p=1):
    """
    args: 
        sample_size:
            Number of nodes to create a graph.
        cluster_num:
            The number of clusters there will be in a given plot.
        shape:
            Shape that will be drawn when plotted, can be blobs, a moons, or circles.
        p:
            Probability of inter-clustering. 
            0 is only connections between a node's own cluster.
            1 can have connections with any cluster.
    return:
        g:
            A NetworkX graph.
        cluster_points:
            The points on the plot that are sorted by which cluster they are in.
    """
    kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=10)
    match shape:
        case "make_blobs":
            X, _ = ds.make_blobs(n_samples=sample_size, centers=cluster_num, cluster_std=0.60, random_state=0)
        case "make_moons":
            X, _ = ds.make_moons(n_samples=sample_size, noise=0.1)
        case "make_circles":
            X, _ = ds.make_circles(n_samples=sample_size, factor=0.33, noise=0.1)

    clustering = SpectralClustering(n_clusters=cluster_num, random_state=0, n_init=10) 
    # spectral clustering is needed to get some labels
    # these labels do not affect the plots
    kmeans.fit(X)
    clustering.fit(X)
    cluster_points = [[] for i in range(cluster_num)]
    node_positions = {}
    labels = clustering.labels_
    g = nx.Graph() 
    for i, (x, y) in enumerate(X): # loop through each data point and get the index and (x, y)
        label = labels[i]
        g.add_node(i, pos=(x, y), cluster=label)
        node_positions[i] = (x, y) # set node positions for if we want to view the graph
        cluster_points[label].append(i)
    for label in range(cluster_num): # loops through the entire cluster 
        nodes = cluster_points[label]
        for node in nodes:# loops through each node
            for _ in range(max(1, cluster_num // 2)):
                target = random.choice(nodes)
                if target != node: # connects nodes together and makes the distance between them their weight
                    x1, y1 = node_positions[node]
                    x2, y2 = node_positions[target]
                    dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    g.add_edge(node, target, weight=dist)
    inter_edges_per_cluster = max(1, cluster_num // 2) # you don't want to have more edges than nodes
    for i in range(cluster_num):
        for k in range(inter_edges_per_cluster):
            if random.random() < p: # here if random < p, it will create a connection to a node outside its own cluster
                src = random.choice(cluster_points[i])
                other_clusters = [j for j in range(cluster_num) if j != i] # loops through and adds all the clusters that we aren't in
                tgt_cluster = random.choice(other_clusters)
                tgt = random.choice(cluster_points[tgt_cluster])
                x1, y1 = node_positions[src]
                x2, y2 = node_positions[tgt]
                dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 # similarly, this connects the nodes and considers its weight
                g.add_edge(src, tgt, weight=dist)
    colors = [labels[i] for i in range(len(X))] # colors for if we want to view
    # nx.draw_networkx(
    #     g,
    #     pos=node_positions,
    #     node_color=colors,
    #     node_size=40,
    #     cmap=plt.cm.viridis,
    #     with_labels=False
    # )
    # plt.show() # to view the graphs uncomment
    return g, cluster_points

def create_spectral_graph(sample_size, cluster_num, shape, p=0.33):
    """
    args: 
        sample_size:
            Number of nodes to create a graph.
        cluster_num:
            The number of clusters there will be in a given plot.
        shape:
            Shape that will be drawn when plotted, can be blobs, a moons, or circles.
        p:
            Probability of inter-clustering. 
            0 is only connections between a node's own cluster.
            1 can have connections with any cluster.
    return:
        g:
            A NetworkX graph.
        cluster_points:
            The points on the plot that are sorted by which cluster they are in.
    """
    match shape:
        case "make_blobs":
            X, _ = ds.make_blobs(n_samples=sample_size, centers=cluster_num, cluster_std=0.60, random_state=0)
        case "make_moons":
            X, _ = ds.make_moons(n_samples=sample_size, noise=0.1)
        case "make_circles":
            X, _ = ds.make_circles(n_samples=sample_size, factor=0.33, noise=0.1)
    clustering = SpectralClustering(n_clusters=cluster_num, gamma=10)
    clustering.fit(X)
    labels = clustering.labels_
    cluster_points = [[] for i in range(cluster_num)]
    node_positions = {}
    g = nx.Graph()
    for i, (x, y) in enumerate(X): # loop through each data point and get the index and (x, y)
        label = labels[i]
        g.add_node(i, pos=(x, y), cluster=label)
        node_positions[i] = (x, y) # set node positions for if we want to view the graph
        cluster_points[label].append(i)
    for label in range(cluster_num): # loops through the entire cluster 
        nodes = cluster_points[label]
        for node in nodes: # loops through each node
            for _ in range(max(1, cluster_num // 2)):
                target = random.choice(nodes)
                if target != node: # connects nodes together and makes the distance between them their weight
                    x1, y1 = node_positions[node]
                    x2, y2 = node_positions[target]
                    dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    g.add_edge(node, target, weight=dist)
    inter_edges_per_cluster = max(1, cluster_num // 2) # you don't want to have more edges than nodes
    for i in range(cluster_num): 
        for k in range(inter_edges_per_cluster):
            if random.random() < p: # here if random < p, it will create a connection to a node outside its own cluster
                src = random.choice(cluster_points[i])
                other_clusters = [j for j in range(cluster_num) if j != i] # loops through and adds all the clusters that we aren't in
                tgt_cluster = random.choice(other_clusters)
                tgt = random.choice(cluster_points[tgt_cluster])
                x1, y1 = node_positions[src]
                x2, y2 = node_positions[tgt]
                dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 # similarly, this connects the nodes and considers its weight
                g.add_edge(src, tgt, weight=dist)
    colors = [labels[i] for i in range(len(X))] # colors for if we want to view
    # nx.draw_networkx(
    #     g,
    #     pos=node_positions,
    #     node_color=colors,
    #     node_size=40,
    #     cmap=plt.cm.viridis,
    #     with_labels=False
    # )
    # plt.show() # to view the graphs uncomment
    return g, cluster_points

def create_dense_or_sparse_graph(sample_size, cluster_num, shape): 
    """
    args: 
        sample_size:
            Number of nodes to create a graph.
        cluster_num:
            The number of clusters there will be in a given plot.
        shape:
            Shape that will be drawn when plotted, can be blobs, a moons, or circles.

    return:
        g:
            A NetworkX graph.
        cluster_points:
            The points on the plot that are sorted by which cluster they are in.
    """
    if argv[4] == "d": # take 5th parameter from command line
        g = nx.dense_gnm_random_graph(30, random.randint(100, 200)) #435=30c2 but 435 really takes a long time
    elif argv[4] == "s":
        g = nx.dense_gnm_random_graph(30, random.randint(30, 100))
    match shape:
        case "make_blobs":
            X, _ = ds.make_blobs(n_samples=sample_size, centers=cluster_num, cluster_std=0.60, random_state=0)
        case "make_moons":
            X, _ = ds.make_moons(n_samples=sample_size, noise=0.1)
        case "make_circles":
            X, _ = ds.make_circles(n_samples=sample_size, factor=0.33, noise=0.1)
    clustering = SpectralClustering(n_clusters=cluster_num, gamma=10)
    clustering.fit(X)
    labels = clustering.labels_
    cluster_points = [[] for i in range(cluster_num)]
    for i, (x, y) in enumerate(X):
        label = labels[i]
        g.add_node(i, pos=(x, y), cluster=label)
        cluster_points[label].append(i)
    return g, cluster_points 
# similar to the other methods, this must also return a graph, and cluster 
# points, otherwise, the quality methods would need to be edited

# here is where I declare all of the lists you may be using in your tests
graphs = []
graphs_for_storage_tests = []
graphs_for_difficulty_tests = []
intervals = [i for i in np.arange(0.001, 0.05, 0.001)] # need to use np.arange() since range() doesn't support floats
print(argv)

match argv[1]: # argv[1] = gets plot type
    case "q/t":
        for i in range(len(intervals)):
            if argv[2] == "k-m": # argv[2] = cluster type
                if len(argv) == 4:
                    graphs.append(create_k_means_graph(30, 5, argv[3], 1)) #300 works for L, doesn't for G
                else:
                    graphs.append(create_dense_or_sparse_graph(30, 5, argv[3])) #argv[3] = shape
            elif argv[2] == "sc":
                graphs.append(create_spectral_graph(30, 5, argv[3], 1))
                if len(argv) == 4:
                    graphs.append(create_spectral_graph(30, 5, argv[3], 1))
                else:
                    graphs.append(create_dense_or_sparse_graph(30, 5, argv[3]))
    case "c/t":
        for i in intervals:
            graphs.append(create_k_means_graph(30, 5, argv[2], 1))
    case "KB/g":
        for i in range(2, len(intervals), 1): 
            graphs_for_storage_tests.append(create_k_means_graph(2*i, 4, argv[3], 1)) # increases node size by 2 each time
    case "q/gd":
        for i in np.arange(1, 0, -.01):
            graphs_for_difficulty_tests.append(create_k_means_graph(30, 5, argv[3], i)) # i = p

def louvain_quality(intervals):
    """
    args:
        intervals:
            The time intervals for which funcitons should recalculate.
    return:
        louvain_quality:
            A list of the quality which was computed using metrics.rand_score() from sklearn.
        louvain_execution_time:
            A list of number of communities after a certain amount of time.
        louvain_memory:
            A list of the amount of storage it takes to compute the communities for a graph.
        louvain_difficulty:
            A list of the quality where the graphs were in increasing difficulty.
    """
    louvain_quality = []
    louvain_communities = []
    louvain_execution_time = []
    louvain_memory = []
    louvain_difficulty = []
    for time in intervals:
        if argv[1] != "q/t":
            break
        s, clusters_ = graphs[intervals.index(time)] # get the graph based on which test we are doing, q/t = graphs[]
        c = louvain_algorithm(s)
        louvain_communities.append(c.get('communities')) # gets the communities 
        louv_ = list(c.get('communities').items()) # k, v pairs
        compare_true =  [None for i in range(len(s.nodes()))]
        compare_pred = [None for i in range(len(s.nodes()))]
        for l in clusters_:
            for i in l:
                compare_true[i] = clusters_.index(l) # turn communities into a format rand_score can use
        for key, value in louv_:
            for i in value:
                compare_pred[i] = key # same idea here
        compare_score = metrics.rand_score(compare_true, compare_pred) # this computes it
        louvain_quality.append(compare_score) 
        print("computing quality results for louvain...")

    for time in intervals:
        if argv[1] != "c/t":
            break
        s, clusters_ = graphs[intervals.index(time)] # get the graph for c/t tests which is graphs[] too
        c = louvain_algorithm(s, time_limit=time)
        if "graph_at_time_limit" in c:
            louvain_execution_time.append(len(c.get("graph_at_time_limit").get("communities"))) # get the communitites part of the dictonary
        else:
            louvain_execution_time.append(len(c.get("communities"))) # if the cutoff is after the algorithm is done, it does not return a "graph_at_time_limit"
        print(f"computing t interval results and storage results for louvain...")

    for i in graphs_for_storage_tests:
        if argv[1] != "KB/g":
            break
        s, clusters_ = graphs_for_storage_tests[graphs_for_storage_tests.index(i)] # storage tests need their own graph because they differ in size
        c = louvain_algorithm(s)
        louvain_memory.append(c.get("memory_used_kb"))
        print("computing memory results for louvain...")
    
    for i in graphs_for_difficulty_tests:
        if argv[1] != "q/gd":
            break
        s, clusters_ = graphs_for_difficulty_tests[graphs_for_difficulty_tests.index(i)]
        c = louvain_algorithm(s)
        louvain_communities.append(c.get('communities'))
        louv_ = list(c.get('communities').items())
        compare_true =  [None for i in range(len(s.nodes()))] # here we do the same thing we did for quality test, but with a different set of graphs
        compare_pred = [None for i in range(len(s.nodes()))]
        for l in clusters_:
            for i in l:
                compare_true[i] = clusters_.index(l)
        for key, value in louv_:
            for i in value:
                compare_pred[i] = key
        compare_score = metrics.rand_score(compare_true, compare_pred)
        louvain_difficulty.append(compare_score)
        print("computing difficulty results for louvain...")
    
    match argv[1]:
        case "q/t":
            return louvain_quality
        case "c/t":
            return louvain_execution_time
        case "KB/g":
            return louvain_memory
        case "q/gd":
            return louvain_difficulty

def girvan_quality(intervals):
    """
    args:
        intervals:
            The time intervals for which funcitons should recalculate.
    return:
        girvan_quality:
            A list of the quality which was computed using metrics.rand_score() from sklearn.
        girvan_execution_time:
            A list of number of communities after a certain amount of time.
        girvan_memory:
            A list of the amount of storage it takes to compute the communities for a graph.
        girvan_difficulty:
            A list of the quality where the graphs were in increasing difficulty.
    """
    girvan_quality = []
    girvan_communities = []
    girvan_execution_time = []
    girvan_memory = []
    girvan_difficulty = []
    for time in intervals:
        if argv[1] != "q/t":
            break
        s, clusters_ = graphs[intervals.index(time)]
        c = GirvanNewman(s, len(clusters_)) 
        comm = c[0] # the c[0] part are the communities
        girvan_communities.append(comm)
        girv_ = comm
        for key, value_set in comm.items(): # k, v pairs
            girv_[key] = list(value_set) # convert set to a list
        compare_true =  [None for i in range(len(s.nodes()))]
        compare_pred = [None for i in range(len(s.nodes()))]
        for l in clusters_:
            for i in l:
                compare_true[i] = clusters_.index(l) # set up in a format for metric.rand_score
        for key, value in girv_.items():
            for i in value:
                compare_pred[i] = key # set up in a format for metric.rand_score
        compare_score = metrics.rand_score(compare_true, compare_pred)
        girvan_quality.append(compare_score)
        print("computing quality results for Girvan-Newman...")

    for time in intervals:
        if argv[1] != "c/t":
            break
        s, clusters_ = graphs[intervals.index(time)]
        c = GirvanNewman(s, len(clusters_))
        exec_time = len(c[0]) # the leght of c[0], that is, len(keys) is the number of communities
        girvan_execution_time.append(exec_time)
        print("computing t interval results for Girvan-Newman...")

    for i in graphs_for_storage_tests:
        if argv[1] != "KB/g":
            break
        s, clusters_ = graphs_for_storage_tests[graphs_for_storage_tests.index(i)] # storage tests have their own graph
        c2 = GirvanNewman_T(s, time)
        memory_used = c2[2] # c2[2] is the memory part
        girvan_memory.append(memory_used)
        print("computing memory results for Girvan-Newman...")

    for i in graphs_for_difficulty_tests:
        if argv[1] != "q/gd":
            break
        s, clusters_ = graphs_for_difficulty_tests[graphs_for_difficulty_tests.index(i)] # difficulty tests have their own graph
        c = GirvanNewman(s, len(clusters_))
        comm = c[0] # c[0] is the dicitonary of communities
        girvan_communities.append(comm)
        girv_ = comm
        for key, value_set in comm.items():# k, v pairs
            girv_[key] = list(value_set)
        print(girv_)
        compare_true =  [None for i in range(len(s.nodes()))]
        compare_pred = [None for i in range(len(s.nodes()))]
        for l in clusters_:
            for i in l:
                compare_true[i] = clusters_.index(l) # set up in a format for metric.rand_score
        for key, value in girv_.items():
            for i in value:
                compare_pred[i] = key # set up in a format for metric.rand_score
        print("compare true", compare_true)
        print("compare pred", compare_pred)
        compare_score = metrics.rand_score(compare_true, compare_pred)
        girvan_difficulty.append(compare_score)
        print("computing difficulty results for Girvan-Newman...")
    match argv[1]:
        case "q/t":
            return girvan_quality
        case "c/t":
            return girvan_execution_time
        case "KB/g":
            return girvan_memory
        case "q/gd":
            return girvan_difficulty

def infomap_quality(intervals):
    infomap_quality = []
    infomap_communities = []
    infomap_execution_time = []
    infomap_memory = []
    infomap_difficulty = []
    for time in intervals:
        if argv[1] != "q/t":
            break
        s, clusters_ = graphs[intervals.index(time)]
        map = Infomap(s, weight=False) # initalize object
        c = map.run()
        comm = c[0]
        infomap_communities.append(comm)
        compare_true =  [None for i in range(len(s.nodes()))]
        info_ = comm
        compare_pred = [None for i in range(len(s.nodes()))]
        for l in clusters_:
            for i in l:
                compare_true[i] = clusters_.index(l) # set up in a format for metric.rand_score
        for key, value in info_.items():
            for i in value:
                compare_pred[i] = key # set up in a format for metric.rand_score
        compare_score = metrics.rand_score(compare_true, compare_pred)
        infomap_quality.append(compare_score)
        print("computing quality results for Infomap...")

    for time in intervals:
        if argv[1] != "c/t":
            break
        s, clusters_ = graphs[intervals.index(time)]
        map = Infomap(s, False) # initalize object
        c = map.run()
        infomap_execution_time.append(len(c[0]))
        print("computing t interval results for Infomap...")

    for i in graphs_for_storage_tests:
        if argv[1] != "KB/g":
            break
        s, clusters_ = graphs_for_storage_tests[graphs_for_storage_tests.index(i)] # storage tests have their own graph
        map = Infomap(s, False) # initalize object
        c = map.run()
        memory_used = c[2]
        infomap_memory.append(memory_used)
        print(f"computing memory results for Infomap... graph {graphs_for_storage_tests.index(i)}/{len(graphs_for_storage_tests)}")

    for i in graphs_for_difficulty_tests:
        if argv[1] != "q/gd":
            break
        s, clusters_ = graphs_for_difficulty_tests[graphs_for_difficulty_tests.index(i)] # difficulty tests have their own graph
        map = Infomap(s, weight=False) # initalize object
        c = map.run()
        comm = c[0]
        infomap_communities.append(comm)
        compare_true =  [None for i in range(len(s.nodes()))]
        info_ = comm
        compare_pred = [None for i in range(len(s.nodes()))]
        for l in clusters_:
            for i in l:
                compare_true[i] = clusters_.index(l) # set up in a format for metric.rand_score
        print(info_)
        for key, value in info_.items():
            for i in value:
                compare_pred[i] = key # set up in a format for metric.rand_score
        compare_score = metrics.rand_score(compare_true, compare_pred)
        infomap_difficulty.append(compare_score)
        print("computing difficulty results for Infomap...")

    match argv[1]:
        case "q/t":
            return infomap_quality
        case "c/t":
            return infomap_execution_time
        case "KB/g":
            return infomap_memory
        case "q/gd":
            return infomap_difficulty

# Here we run all the tests based off what you entered when you ran the program
Louvain = louvain_quality(intervals)
Girvan = girvan_quality(intervals)
Info = infomap_quality(intervals)
title = argv[1] + " " + argv[2] # title becomes what you passed in 
match argv[1]:
    case "q/t":
        x = intervals
        title += " " + argv[3]
        if len(argv) == 5: title += " " + argv[4] # in the case you put "d or "s" 
        plt.xlabel("time (s)")
        plt.ylabel("quality score")
    case "c/t":
        x = intervals
        plt.xlabel("time (s)")
        plt.ylabel("number of communities")
    case "KB/g":
        x = np.arange(0, len(graphs_for_storage_tests), 1).tolist()
        plt.xlabel("graph size (nodes)")
        plt.ylabel("storage used (KB)")
    case "q/gd":
        x = np.arange(1, 0, -.01).tolist() 
        plt.xlabel("difficulty (p=cluster independence)")
        plt.ylabel("quality score")
# standard plotting 
plt.title(title)
plt.scatter(x, Louvain, label="Louvain")
plt.scatter(x, Girvan, label="Girvan-Newman")
plt.scatter(x, Info, label="Infomap")
plt.legend()
plt.show()