import numpy as np

class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.node_size = len(nodes)
        self.edge_size = len(edges)
        self.node_name_to_node_index_dict = {}
        self.matrix = np.zeros((self.node_size, self.node_size), dtype=int)
        
        for i in range(self.node_size):
            self.node_name_to_node_index_dict[nodes[i]] = i

        for edge in edges:
            self.add_edge(edge)
        
    def add_node(self, node): # adds a new node to the adjacency matrix, pass in the name of the node. ex: "v0"
        new_row = np.zeros(self.matrix[0].size, dtype=int)
        self.matrix = np.append(self.matrix, [new_row], axis=0)
        self.nodes.append(node)
        self.node_size += 1
        self.matrix = np.insert(self.matrix, len(self.matrix[0]), [0], axis=1)
        self.node_name_to_node_index_dict[node] = len(new_row) #np.where(self.matrix == node)
        #print(self.node_name_to_node_index_dict[node])

    def remove_node(self, node): # removes a node from the adjacency matrix, pass in the name of the node. ex: "v0"
        self.matrix = np.delete(self.matrix, self.nodes.index(node), axis=0)
        self.matrix = np.delete(self.matrix, self.nodes.index(node), axis=1) #self.nodes.index(node))
        self.nodes.pop(self.nodes.index(node))

    # add an edge tuple to the matrix, in 1 direction, from first value to latter value. 
    # ex: ("v1", "v2") y axis of matrix is "from" node, x axis is "to" node
    # optionally in the tuple you can include the weight: ("v1", "v2", 5)
    def add_edge(self, edge): 
        if len(edge) == 2: # no weight
            node_from = self.node_name_to_node_index_dict[edge[0]]
            node_to = self.node_name_to_node_index_dict[edge[1]]
            self.matrix[node_from, node_to] = 1
        elif len(edge) == 3: # with weight
            node_from = self.node_name_to_node_index_dict[edge[0]]
            node_to = self.node_name_to_node_index_dict[edge[1]]
            self.matrix[node_from, node_to] = edge[2]

    def remove_edge(self, edge): # to remove an edge, provide the edge tuple ex: ("v1", "v2")
        node_from = self.node_name_to_node_index_dict[edge[0]]
        node_to = self.node_name_to_node_index_dict[edge[1]]
        self.matrix[node_to, node_from] = 0

    def get_matrix(self): # returns the matrix
        return self.matrix

    def get_nodes(self): # returns a list of the vertex names
        return self.nodes