# CP307

### Quality Analysis Tests

This file provides a series of tests that are based off the **Louvain Method**, **Girvan-Newman Method**, and the **Infomap Algorithm** which were implemented in Python using the help of **NetworkX**.  

---
### File Structure
- ```quality_analysis_test.py```
- ```louvain.py```
- ```girvan_newman.py```
- ```infomap.py```

---

## To run
```~ % python quality_analysis_tests.py {test_type}{cluster_method} {shapes} {dense_or_sparse (this is optional ,only use if plot type is q/t)}```
#### test_types 
- ```q/t``` quality vs. time
- ```c/t``` communities vs. time
- ```KB/g``` kilobytes vs. graph size in nodes
- ```q/gd``` quality vs. graph difficulty
#### cluster_methods
- ```k-m``` K-means clustering
- ```sc``` Spectral clustering
#### shapes
- ```make_blobs``` the points in the graph will shape into blobs
- ```make_moons``` the points in the graph will shape into moons
- ```make_circles``` the points in the graph will shape into circles
#### dense_or_sparse
- ```d``` creates a dense graph
- ```s``` creates a sparse graph
##### Example
```~ % python quality_analysis_tests.py q/t k-m make_blobs d```


## To run Louvain 
* Args:
    - ```G```: NetworkX graph
    - ```delta_q```: Minimum modularity gain threshold to accept a move the 
                     smaller the number the easier it is to form communities, 
                     ie. larger communities 
    - ```time_limit```: time limit in seconds to capture intermediate state,   
                        usually no larger than 0.05 seconds for 'small' graphs.
   
* Returns:
    - ```dict```: Contains 'communities', 'execution_time', 'memory_used', 
            and optionally 'graph_at_time_limit' if time_limit is specified will all of the same above information

* example: ```louvain_algorithm(Graph, delta_q=0.5, time_limit=0.05)```


## To build Infomap 
* Args:
    - ```G```: NetworkX graph
    - ```weight```: Boolean indicating if the graph is weighted

* example: ```infomap(G, weight)```

## To run Infomap 
* Args:
    - ```self```: Infomap object
    - ```iters```: The number of outer passes over all nodes
   
* Returns:
    - ```tuple```: Contains 'retval', 'total_time', 'memory_used_kb'

* example: ```infomap.run(iters=10)```

## To run Girvan Newman 
- there are two methods that could be ran to run the whole method
    - GirvanNewman(gh, k) (regular method)
    - GirvanNewman_T(gh, tm) (used to place a time limit for running the algorithm)

* Args: GirvanNewman(gh, k)
    - ```gh```: the networkx graph
    - ```k```: The number of desired components
   
* Returns:
    - ```dict```: Contains 'com', 'total_time', 'memory_used_kb'
 * example: ```GirvanNewman(graph, 10)```
      
* Args: GirvanNewman_T(gh, tm)
    - ```gh```: the networkx graph
    - ```tm```: The time limit for running algorithm
   
* Returns:
    - ```dict```: Contains 'com', 'total_time', 'memory_used_kb'
* example: ```GirvanNewman_T(graph, 0.05)```

  
### Credits go to our wonderful project members :)
