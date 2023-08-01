import numpy  as np
from decimal import *

def create_mat_t(mat, a_tm1, d, retention):
    """
    @params:
        mat: numeric matrix 
        a_tm1: numeric vector
        d: numeric vector
        retention: numeric matrix
    @return:
        mat_t: numeric matrix
    """
    n_row, n_col = mat.shape[0], mat.shape[1]
    mat_t = np.zeros((n_row,n_col)) 
    # the (i,j) entry represents the activation going from i to j.
    for i in range(0, n_row):
        for j in range(0, n_col):
            if i == j:
                mat_t[i,j] = a_tm1[i] if d[i] == 0 else retention[i] * a_tm1[i]
            elif d[i] == 0:
                mat_t[i,j] = 0
            else:
                mat_t[i,j] = (1 - retention[i]) * mat[i,j] * (a_tm1[i]/d[i])
          
    return mat_t