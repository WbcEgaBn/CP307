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
```~ % python quality_analysis_tests.py {test_type}{cluster_method} {shapes} {dense_or_sparse (only if plot type is q/t)}```
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

### Credits go to our wonderful project members :)