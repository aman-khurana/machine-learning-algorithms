import numpy as np
import pandas as pd

def pca(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """

    ### scaling X
    X_scaled = X_train - X_train.mean(axis = 0)

    ### dividing scaled X by standard deviation 
    X_std = X_scaled/np.std(X_scaled, axis=0)


    cov_X = np.dot(X_std.T,X_std)/X_std.shape[0]

    ## eigen value decomposition
    e_val, e_vec = np.linalg.eig(cov_X)

    idx = np.argsort(e_val)[::-1]

    e_vec = e_vec[:,idx]
    e_val = e_val[idx]

    ### transform data

    X_reduced = np.dot(X_std,e_vec[:,:N])

    return X_reduced

if __name__ == "__main__":

    df = pd.read_csv('data.csv')
    
    X = df.values[:,:-1]
    y = df.values[:,-1]

    num_columns = X.shape[1]
    num_components = 5
    print('original num columns', num_columns)
    X_reduced  = pca(X, num_components)
    print('reduced num columns', X_reduced.shape[1])

    