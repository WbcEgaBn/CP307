# CP307

### Graph Class (Adjacency Matrix Implementation)

This class provides a Graph data structure implemented in Python using **NumPy**.  
It supports creating a graph from a list of nodes and edges, and provides operations for adding/removing nodes, adding/removing edges (with optional weights), and retrieving the adjacency matrix.

---

## Class Structure

```python
import numpy as np

class Graph:
    def __init__(self, nodes, edges):
        ...
    def add_node(self, node):
        ...
    def remove_node(self, node):
        ...
    def add_edge(self, edge):
        ...
    def remove_edge(self, edge):
        ...
    def get_matrix(self):
        ...
    def get_nodes(self):
        ...
