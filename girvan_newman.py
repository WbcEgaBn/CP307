#need a graph, maybe that is sent in - we should be working on the same graph - maybe this task can belong to joshua or someone else

import networkx as nx
import random
import time
import tracemalloc
#Girvan Newman:
#compute the edge betweeness
#identify and remove the edge with max betweeness
#recompute the edge betweeness
#

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


def removeEdge(bw_edge, gr, edgs):
    #edges with the most betweenes are removed - but one at a time and then recalculate between
    # but what is removing doing - how does it split?
    bw_max = 0
    edgeIdx = -1
    bt = {}
    
    for i in range(len(bw_edge)):
        

        if bw_edge[edgs[i]] > bw_max:
            bw_max = bw_edge[edgs[i]]
            edgeIdx = i
        
    
    #remove the edge with highest betweeness centrality
    
        
    edg = [k for k,v in bw_edge.items() if v == bw_edge[edgs[edgeIdx]]]
    if len(edg) != 0:
        gr.remove_edge(edg[0][0], edg[0][1]) #right we only want the first edge that has this high score

    #bw_edge.remove(edgeIdx)

    return gr #should we return it back or no???
        #need something to redraw the graph
    

    

def showCommunity(graph):
    #show the communites it has split into
    #use the find neighbors - right?
    #
    coms = nx.connected_components(graph)
    communities = {}
    i =1
    for com in coms:
        
        communities[i] = com
        i = i +1 
        
    return communities
    


def GirvanNewman(gh):
    #where we run the whole thing!
    
    st = time.time()
    tracemalloc.start()
    
    
    p = shortestPath(gh)
    #print(p)
    #^^shortestPath seems to work

    bt, ls = calculateB(p, gh)
    #print(bt)
    #^^ betweeness score works

    gh = removeEdge(bt, gh, ls)

    #print(nx.is_connected(h))
    #could do it until no longer connected
    #nx.is_connected(h)
    k = 5
    while k > 0:
        
        p = shortestPath(gh)
    #print(p)
    #^^shortestPath seems to work

        bt, ls = calculateB(p, gh)
    #print(bt)
    #^^ betweeness score works

        gh = removeEdge(bt, gh, ls)
        k = k - 1

    et =  time.time()
    tTotal = et - st

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_used_kb = peak / 1024

    return [showCommunity(gh), tTotal, memory_used_kb]
        


if __name__ == "__main__":
    
    
    nds = list(range(15))
    #nds = random.shuffle(nds)

    for i in range(len(nds)):
        t = nx.erdos_renyi_graph(nds[i], 0.5)
        f, t = GirvanNewman(t)
        print(f, "time: ", t)

