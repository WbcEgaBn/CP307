import networkx as nx
import random
from matplotlib import pyplot as plt
from louvain import louvain_algorithm
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


create_random_graph(10, 1, 1)

input_sparse_graphs = []
input_dense_graphs = []
for i in range(500):
    node_num = random.randint(0, 1000)
    g = create_random_graph(random.randint(0, 1000), 1, 1)#node_num%2)
    input_sparse_graphs.append(g)
for i in range(500):
    node_num = random.randint(0, 1000)
    g = create_random_graph(random.randint(0, 1000), 3, 5)
    input_dense_graphs.append(g)
output_sparse_alg_1 = []
output_dense_alg_1 = []
for i in range(500):
    output_sparse_alg_1.append(louvain_algorithm(input_sparse_graphs[i]))
    output_dense_alg_1.append(louvain_algorithm(input_dense_graphs[i]))
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

