import math
import random
import time
import tracemalloc

class Infomap():
    
    # initialization of the infomap where G is a nx array and weight is a boolean
    
    def __init__(self, G, weight):
        self.G = G
        self.weight = bool(weight)
        self.nodes = list(self.G.nodes())
        self.edges = list(self.G.edges())
        self.sz = len(self.nodes)


    # get the total edge weight for the graph
    # unweighted: returns number of edges
    # weighted: returns sun of edge weights
    
    def get_edge_weights(self):
        if self.weight == True:
            total_weight = 0.0
            for i, j, k in self.G.edges(data=True):
                total_weight += float(k.get("weight", 1.0)) # get the weight val of an edge (from edge dict), default to 1 if it does not exist
            return total_weight
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
                probabilities[n] = float(s) / (2.0 * total)
        # if unweighted
        else:
            degrees = dict(self.G.degree())  # get dict of neighbors
            for n in self.nodes:
                d = degrees.get(n, 0) # get for node n
                probabilities[n] = float(d) / (2.0 * total)
                
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
                w = float(k.get("weight", 1.0))
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

        # fix so that community keys are in the correct order
        ordered = sorted(communities.keys())
        remap = {com: i for i, com in enumerate(ordered)}
        newindex = {remap[i]: communities[i] for i in ordered}
        return newindex


    # compute l: the per-step description length for module partition M. 
    # that is, for module partition M of n nodes into m modules
    # L = q * H(Q) + sum from m=1 to n of m pdot m * H(p of m)
    # q = sum q_m (the total probability that the random walker enters any of the m modules)
    # pdot m = q_m + sum for i in m of pi[i] (which is given by the total
    # probability that any node in the module is visited, plus the probability that the
    # random walker exits the module and the exit codeword is used
    # H(Q) = q_m / q (The frequency-weighted average length of codewords in the index codebook)
    # H(p of m) = q_m and pi[i] for in in m over pdot m (The entropy of the relative rates at which 
    # the random walker exits module i and visits each node in module i
    
    def compute_l(self, partition, edge_weights=None, pi=None, q_m=None):

        # set initial values
        if edge_weights is None:
            edge_weights = self.get_edge_weights()
    
        if pi is None:
            pi = self.get_stationary_probability()
    
        if q_m is None:
            q_m = self.get_exit_probabilities(partition)
    
        modules = set(partition.values())
    
        # initialize variables
        q = 0.0
        p = 0.0
        H_Q = 0.0
        H_Pm = 0.0
    
        # compute q
        for m in modules:
            q += q_m.get(m, 0.0)
    
        # compute H(Q)
        if q > 0.0:
            H_Q = -sum(
            (q_m[m]/q) * math.log2(q_m[m]/q)
            for m in modules
            if q_m[m] > 0.0
        )
    
        # compute H(p of m)
        for m in modules:
            # compute p dot m (how often we are in community m)
            prob_of_visit = []
            for i in self.nodes:
                if partition[i] == m:
                    prob_of_visit.append(pi[i])
    
            prob_of_exit = q_m[m]
            pdot = sum(prob_of_visit) + prob_of_exit
    
            if pdot > 0:
                probs = []
                probs.append(prob_of_exit / pdot)
    
                for i, comm in partition.items():
                    if comm == m:
                        probs.append(pi[i] / pdot)
    
                H_Pm += pdot * (-sum(p * math.log2(p) for p in probs if p > 0))
    
        return q * H_Q + H_Pm
    

    # run infomap
    
    def run(self):
        # start timer
        start_time = time.time()

        # start memory tracking
        tracemalloc.start()

        # build starting communities
        communities = {n: i for i, n in enumerate(self.nodes)}
        pi = self.get_stationary_probability()

        # compute initial L
        L = self.compute_l(communities, pi=pi)

        # number of iterations (can change as needed)
        for i in range(10):
    
            # randomize order of nodes
            node_list = list(self.nodes)
            random.shuffle(node_list)
    
            for n in node_list:
                # get the current community of n
                current = communities[n]
    
                # find the community ids of n's neighbors
                neighbor_comms = set()
                for ni in self.G.neighbors(n):
                    neighbor_comms.add(communities[ni])
    
                # initialize tracking metrics
                current_L = L
                new_current = current
                best_L = current_L

                # move n into the community of its neighbors
                for node in neighbor_comms:
                    communities[n] = node
                    # compute new L for node
                    new_L = self.compute_l(communities, pi=pi)

                    # if the new description length is lower we update
                    # the best description length
                    if new_L < best_L:
                        best_L = new_L
                        new_current = node

                # if we have found a better move
                if new_current != current:
                    # move the node into that community and update L
                    communities[n] = new_current
                    L = best_L
                else:
                    communities[n] = current

        # map return values to the correct format
        retval = self.map(communities)

        # end timer
        end_time = time.time()
        total_time = end_time - start_time

        # end memory tracking
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # convert units
        memory_used_kb = peak / 1024
        
        return [retval, total_time, memory_used_kb]
    
