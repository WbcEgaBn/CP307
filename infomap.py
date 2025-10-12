import numpy as np
import networkx as nx

class Infomap():
    # initialization of the infomap where G is a nx array and weight is a boolean
    
    def __init__(self, G, weight):
        self.G = G
        self.weight = bool(weight)
        self.nodes = list(self.G.nodes())
        self.edges = list(self.G.edges())
        self.sz = len(self.nodes)

    
    # build starting communities where each node is it's own community
    
    def build_starting_communities(self):
        communities = {}
        for i, n in enumerate(self.nodes):
            communities[i] = [n]   # each community contains a list of nodes
        return communities


    # get the total edge weight for the graph
    # unweighted: returns number of edges
    # weighted: returns sun of edge weights
    
    def get_edge_weights(self):
        if self.weight == True:
            total_weight = 0.0
            for i, j, k in self.G.edges(data=True):
                total_weight += float(k.get("weight", 1.0)) # get the weight val of an edge (from edge dict), default to 1 if it does not exist
        else:
            return float(self.G.number_of_edges())


    # get stationary probalility for moving between nodes in a random walk
    # i.e. fraction of time spent at this node
    # uw: pi[n] = degree[n] / (2m)    - number of edges connected to n divided bu 2 times total num edges
    # w: pi[n] = strength[n] / (2w)   - sum of weights connected to n divided by 2 times total edge weights
    
    def get_stationary_probability(self):
        total = self.get_edge_weights()  # m or w
        probabilities = {}
    
        # if there are no edges return 0
        if total == 0:
            for n in self.nodes:
                probabilities[n] = 0.0
            return probabilities
    
        # if weighted
        if self.weight == True:
            strengths = dict(self.G.degree(weight="weight"))  # get dict of neighbors
            for n in self.nodes:
                s = strengths.get(n, 0.0) # get for node n and cast to float
                probabilites[n] = float(s) / (2.0 * total)
        # if unweighted
        else:
            degrees = dict(self.G.degree())  # get dict of neighbors
            for n in self.nodes:
                d = degrees.get(n, 0) # get for node n
                probabilites[n] = float(d) / (2.0 * total)
                
        return probabilities


    # get exit probabilities
    # qm = weight of edges from u to k / 2 * weight of edges total (divide by 2 so it sums to a prob (1))
    # node_communities = dict from node to community id
    
    def get_exit_probabilities(self, node_communities):
        total_weight = self.get_edge_weights()
        
        # if total weight = 0 return 0
        if total_weight == 0:
            return {c: 0.0 for c in set(node_communities.values())}
    
        q_m = {c: 0.0 for c in set(node_communities.values())}
    
        # loop over edges
        # if weighted
        if self.weight == True:
            for i, j, k in self.G.edges(data=True):
                w = float(data.get("weight", 1.0))
                # if in diff communities count it, if not, skip
                if node_communities[i] != node_communities[j]:
                    q_m[node_communities[i]] += w
                    q_m[node_communities[j]] += w
        # if unweighted
        else:
            for i, j in self.G.edges():
                # if in diff communities count it, if not, skip
                if node_communities[i] != node_communities[j]:
                    q_m[node_communities[i]] += 1.0
                    q_m[node_communities[j]] += 1.0
    
        # normalize so it sums to a prob
        for c in q_m:
            q_m[c] /= (2.0 * total_weight)
    
        return q_m


    # map final communities to a dictionary to return
    
    def map(self, node_comms):
        communities = {}
        for n, c in node_comms.items():
            if c not in communities:
                communities[c] = []
            communities[c].append(n)
        return communities



    # compute l implement equation to group/partition communities
    
    def compute_l(self):
        pass


    # run to test
    
    def run(self):
        node_comms = {n: i for i, n in enumerate(self.nodes)}
        return self.map(node_comms)
    
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3)])

infomap = Infomap(G, weight=False)
result = infomap.run()
print(result)