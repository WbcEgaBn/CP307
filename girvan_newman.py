import networkx as nx

import time
import datetime
import tracemalloc

#### Girvan Newman:
## 1. compute the edge betweeness
## 2. identify and remove the edge with max betweeness
## 3. recompute the edge betweeness
#NOTE: Due to my own version of the girvan newman being extremely slow, this code only utilizes the package
# networkx has for bwtweeness centrality. Therfore, some of my methods are commented out for timing sake but have been tested in jupyter.


"""
class QBag():
    def __init__(self):
        self.n = []

    def add(self, item):
        self.n.append(item)

    
    def rm(self):
        return self.n.pop(0)
    
    def ifempty(self):
        return len(self.n) == 0
    
    def empty(self):
        while not self.ifempty():
            self.n.pop()
    
class PQBag():
    def __init__(self):
        self.cp = []
    def add(self, item, p):
        
        #check if item already exist - then change its priority
        for i in range(len(self.cp)):
            if self.cp[i][0] == item:
                if p < self.cp[i][1]:
                    self.cp[i] = (item, p)
                    self.heapProp(i+1)

        if (item, p) not in self.cp:

            self.cp.append((item, p))
            self.heapProp(len(self.cp))
        
    def take(self):
        temp = self.cp[0]
        self.cp[0] = self.cp[len(self.cp)-1]
        self.cp.pop(len(self.cp)-1) # pop out the last item
        
        self.heapProp(1)
        return temp

    def heapProp(self, sz):
        min = 0
        #should i change this to just look at the next node instead of parent -like a queue?
        
        #if statement to see whwere we are starting
        if sz == len(self.cp):
             #checking up - placing the added node in its right place
            while sz // 2 > 0:
            
                if self.cp[sz-1][1] < self.cp[(sz // 2)-1][1]: #index for parent 
                    temp = self.cp[(sz//2)-1]
                    self.cp[(sz//2) -1] = self.cp[(sz-1)]
                    self.cp[(sz-1)] = temp

                sz = sz //2
             
        else:
             #checking down - placing the nodes in its right place
            while (sz * 2) <= len(self.cp):
                min = self.min(sz)
                if self.cp[sz-1][1] > self.cp[min-1][1]:
                    temp =  self.cp[sz-1]
                    self.cp[sz-1] = self.cp[min-1]
                    self.cp[min-1] =  temp

                sz = min

                
    def min(self, sz):
        if ((sz*2) +1) > len(self.cp):
            return sz * 2 #return left child index
        else:
            if self.cp[(sz*2)-1][1] < self.cp[((sz*2) +1)-1][1]:
                return sz * 2
            else:
                return sz * 2 + 1
            

    def empty(self):
        return len(self.cp) == 0

##dikstra algorithm to find the shortest path between all node and each other
#takes in g - graph, u- starting node, v - ending node, qb - bag
#returns the edge paths
def dijkstra(g, u,v, qb):
    qb.add(u, 0)
    
    visited = []
    preds = {}
    best_node = None
    
    # as long as there are still things in the bag:
    while not qb.empty():
        # take something out of the bag
        cur = qb.take()
        
        # if we've already visited this thing, pass
        if cur[0] in visited:
            continue
            
        # if we have found the node we're looking for, end the loop
        if cur[0] == v:
            
            break
            
        # if neigther of those two cases apply, get the neighbors
        neigh = list(g.neighbors(cur[0]))
        
        # add each neighbor to the bag
        for n in neigh:
            #add if statement for if it has block or body - don't add to bag
            
            p = abs(((u - n)**2)**.5)

            qb.add(n, p)
            if n not in visited:
                

                
                if n in preds:
                    tmp = preds[n]
                    
                    if tmp[1] > cur[1]:
                        preds[n] = cur
                    
                else:
                    preds[n] = cur
                
            if n != v:
                
                if not qb.empty():
                    if best_node is None:
                            best_node = (n, p)
                            
                    else:
                        if best_node[1] > p:

                            best_node = (n,p)
                            preds[best_node] = cur
            else:
                if best_node is None:
                    best_node = (u,0)

                preds[v] = best_node
        best_node = None
        # mark the current node as visited
        visited.append(cur[0])
        
        
    path = [cur[0]]
    p_eds = []
    
    while cur[0] != u:
        if cur[0] in preds:
            pre = preds[cur[0]]
            suc = cur[0]
            if g.has_edge(pre[0],suc):
                
                p_eds.append((pre[0],suc))
            cur = preds[cur[0]]
            
            path.append(cur[0])
    path.reverse()
    
    return p_eds

##dikstra algorithm to find the shortest path between all node and each other
#takes in g - graph, u- starting node, v - ending node, qb - bag
#returns the edge paths
def CFS(g, u,qb):

    qb.add(u)
    visited = []
    preds = {}
    found = False

    while not qb.ifempty():

        cur = qb.rm()

        if cur not in visited:
           
            neighbors  = list(g.neighbors(cur))


            for n in neighbors:
                qb.add(n)
                if n not in visited:
                    preds[n] = cur


            visited.append(cur)


    path = [cur]
    
    while cur!= u:
        
        
        cur = preds[cur]
        path.append(cur)

    path.reverse()
    
    return path






## shortestPath  method utilizes the dijkstra's algorithm
#takes in gr - graph
# returns all possible short paths
def shortestPath(gr):
    #wait till class time to start this part
    #breadth first search BFS
    #return a dic/tuple of node with shortes path???
    b = PQBag()
    paths = []
    ls = list(gr.nodes)
    
    
    for n in ls:

        for nt in ls:
            
            paths.append(dijkstra(gr, n, nt, b))
            

    return paths


def calculateB(shrtLst, g):
    # each node should have been assign a betweeness centrality score - should that be
    
    egs = list(g.edges)

    bwt_cen = {}

    #betweeness centrality = the number of paths that pass through v // total # of shortest from node a to node b
    #create a dict of each node and its betweeness
    for e in egs:
        bwt = 0
     
        occur = 0
        
        #loop through each node and calculate
        for sht in shrtLst:
            
            if e in sht:
                
                occur = occur +  1
        
        bwt = occur / len(shrtLst)
        
        
        bwt_cen[e] = bwt   

    
     
    return bwt_cen

    
## this shows the communities in the graph by using BFS
#takes in graph
#returns the communities - dict{}
def showCommunity(graph):
    #show the communites it has split into
    #use the find neighbors - right?
    
    communities = {} 
    #USE a DFS and whatever node was never visited at the end of this will probably be its own community
    #the path will be the community we use
    b = QBag()
    l = list(graph.nodes)
    
    #need_vis = []
    com_idx = 1
    while len(l) != 0:
        cm = []
        
        p = CFS(graph, l[0],b)
        
        for n in l:
            if n in p:
                
                cm.append(n)
                
                l.remove(n)
                
        communities[com_idx] = cm
                
        
        com_idx = com_idx +1

            
    return communities

"""


#### removeEdge ####
# this method partitions th graph by removing the edge with the highest centrality each time
# it is called
# bw_edge: a dictionary of edges(key) and their betweeness centrality (value)
# gr: the networkx graph
# returns: gr
def removeEdge(bw_edge, gr):
    #edges with the most betweenes are removed - but one at a time and then recalculate between
   
    bw_max = 0
    edgeIdx = -1
    
    
    for bw in bw_edge:
        

        if bw_edge[bw] > bw_max:
            bw_max = bw_edge[bw]
            edgeIdx = bw
    
    #remove the edge with highest betweeness centrality
    
        
    edg = [k for k,v in bw_edge.items() if v == bw_edge[edgeIdx]]
    if len(edg) != 0:
        gr.remove_edge(edg[0][0], edg[0][1]) #we only want the first edge that has this high score


    return gr #return the partitoned graph back
    

    
#### showCommunity ####
# this method creates a dictionary of the communities based on of the connected 
# components in the graph
# graph: the networkx graph
#  communities: a dictionary of communities(key) and their respective nodes (value)
# returns: communities
def showCommunity(graph):
    #show the communites/components it has split into
    #
    coms = nx.connected_components(graph)
    communities = {}
    i =1
    for com in coms:
        
        communities[i] = com
        i = i +1 
        
    return communities
    

#### GirvanNewman Method: ####
# this is where all the code gets ran with NO time limit
# gh: the networkx graph
# k: the number of components desired??
# com: a dictionary of communities(key) and their respective nodes (value)
# tTotal:  the total amount of time to run the code in seconds (float)
# mem: the amount of memory used to run the code
# return: com, tTotal, mem
def GirvanNewman(gh, k):
    #where we run the whole thing!
    
    st = time.time()
    tracemalloc.start()
    
    
    
    bt = nx.edge_betweenness_centrality(gh)
    gh = removeEdge(bt, gh)

    
    
    cps = nx.number_connected_components(gh) #num of components currently

    while k > cps:
        
        bt = nx.edge_betweenness_centrality(gh) 

        gh = removeEdge(bt, gh)
        
        

        cps = nx.number_connected_components(gh)


    #what to return:
    et =  time.time()
    tTotal = et - st
    com = showCommunity(gh)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_used_kb = peak / 1024
    
    return [com, tTotal, memory_used_kb]
        

#### GirvanNewman Method: ####
# this is where all the code gets ran but with a TIME LIMIT
# gh: the networkx graph
# tm: time limit to run the algorithm ***
# com: a dictionary of communities(key) and their respective nodes (value)
# tTotal:  the total amount of time to run the code in seconds (float)
# mem: the amount of memory used to run the code
# return: com, tTotal, mem
def GirvanNewman_T(gh,tm):
    #where we run the whole thing!
    
    st = time.time()
    tracemalloc.start()
    
    
    
    bt = nx.edge_betweenness_centrality(gh)
   
    gh = removeEdge(bt, gh)

    
    
    cur_time = time.time() - st
    while cur_time < tm:
        
        bt = nx.edge_betweenness_centrality(gh) 

        gh = removeEdge(bt, gh)
        

        cur_time = time.time() - st
    #what to return:
    et =  time.time()
    tTotal = et - st
    com = showCommunity(gh)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_used_kb = peak / 1024

    return [com, tTotal, memory_used_kb]

