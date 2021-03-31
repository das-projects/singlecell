from memory_profiler import profile
import numpy as np
from scanpy import neighbors

np.random.seed(0)
X = np.random.rand(20000, 50)


@profile
def numpy():
    neighbors.compute_neighbors_numpy(X, n_neighbors=10)


@profile
def umap():
    neighbors.compute_neighbors_umap(X, n_neighbors=10)


# numpy()
umap()
