import networkx as nx

import time
import datetime
import tracemalloc

#### Girvan Newman:
## 1. compute the edge betweeness
## 2. identify and remove the edge with max betweeness
## 3. recompute the edge betweeness
#NOTE: Due to my own version of the girvan newman being too slow, this code only utilizes the package
# networkx has for bwtweeness centrality. Therfore, some of my methods are commented out for timing sake.


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
    

def BFS(g, u,v, qb):
    qb.add(u)
    visited = []
    preds = {}
    found = False

    while not found and not qb.ifempty():

        cur = qb.rm()

        if cur not in visited:
            if cur == v:
                found == True

            else:
                neighbors  = list(g.neighbors(cur))


                for n in neighbors:
                    qb.add(n)
                    if n not in visited:
                        preds[n] = cur


                visited.append(cur)

        
    path = [cur]
    p_eds = []
    while cur!= u:
        #save the successor
        #save the predecessor
        #check if the edge exist between them (pred, succ)
        pre = preds[cur]
        suc = cur
        if g.has_edge(pre,suc):
            
            p_eds.append((pre,suc))
        
        cur = preds[cur]
        path.append(cur)

    path.reverse()
    qb.empty()
    return p_eds





def shortestPath(gr):
    #wait till class time to start this part
    #breadth first search BFS
    #return a dic/tuple of node with shortes path???
    b = QBag()
    paths = []
    ls = list(gr.nodes)
    
    
    for i in range(len(gr.nodes)):

        for j in range(len(gr.nodes)):
            paths.append(BFS(gr, ls[i], ls[j], b))
            

    return paths


def calculateB(shrtLst, g):
    # each node should have been assign a betweeness centrality score - should that be
    #included in the graph class???
    #betweeness - how frequently it appears in the shortest paths between node pairs 
    #should be calculated after the graph is made 
    #we need a shortest path method - is that in the graph class - BFS
    #we need a dictionary for edges and their betweeness
    egs = list(g.edges)

    bwt_cen = {}

    #betweeness centrality = the number of paths that pass through v // total # of shortest from node a to node b
    #create a dict of each node and its betweeness
    for i in range(len(egs)):
        bwt = 0
        ed = egs[i]
        occur = 0
        
        #loop through each node and calculate
        for j in range(len(shrtLst)):
            
            if ed in shrtLst[j]:
                
                occur = occur +  1
        
        bwt = occur / len(shrtLst)
        
        
        bwt_cen[ed] = bwt   

    
     
    return bwt_cen, egs

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

    
    #nx.is_connected(h)
    #k = ((len(gh.nodes))//(2/3) ) + 4
    #k = (len(gh.nodes)) * 1.5
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

    
    #nx.is_connected(h)
    #k = ((len(gh.nodes))//(2/3) ) + 4
    #k = (len(gh.nodes)) * 1.5
    #cps = nx.number_connected_components(gh) #num of components currently
    #curr_time = datetime.datetime.now()
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

