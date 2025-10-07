from graph import Graph
# this example is in the textbook (8.2) it you want a visual representation
# just like in that example, we will assume that each edge is one directional
graph_nodes = ["v0", "v1", "v2", "v3", "v4", "v5"]
graph_edges = [("v0", "v1"), ("v1", "v2"), ("v2", "v3"), ("v3", "v4"),
               ("v4", "v0"), ("v0", "v5"), ("v5, v4"), ("v3", "v5"), ("v5", "v2")]
g = Graph(graph_nodes, graph_edges)
print(g.get_matrix())
g.remove_node("v4")
print(g.get_matrix())
print(g.get_nodes())