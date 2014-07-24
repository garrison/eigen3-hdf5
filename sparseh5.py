# to be used with h5py

import numpy as np
from scipy.sparse import coo_matrix

def to_sparse_repr(mat):
    mat = coo_matrix(mat)
    dtype = [('r', int), ('c', int), ('v', mat.dtype)]
    m = np.zeros(shape=(mat.nnz,), dtype=dtype)
    m["r"] = mat.row
    m["c"] = mat.col
    m["v"] = mat.data
    return m

def from_sparse_repr(m, shape=None):
    return coo_matrix((m["v"], (m["r"], m["c"])), shape=shape).tocsr()

def save_sparse_h5(grp, name, data):
    grp[name] = to_sparse_repr(data)
    grp[name].attrs["shape"] = data.shape

def load_sparse_h5(grp, name):
    shape = grp[name].attrs["shape"]
    return from_sparse_repr(grp[name], shape=shape)
