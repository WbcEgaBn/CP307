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


### Credits go to our wonderful project members :)
