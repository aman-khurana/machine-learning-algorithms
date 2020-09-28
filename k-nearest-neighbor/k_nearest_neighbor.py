
import numpy as np
from sklearn.preprocessing import StandardScaler

def knn(X_train,X_test,Y_train, k):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    X_combined = np.vstack((X_train, X_test))

    sc = StandardScaler()
    sc.fit(X_combined)

    X_scaled = sc.transform(X_train)
    X_test_s = sc.transform(X_test)
    n = X_train.shape[0]
    
    y_preds = []
    for row in X_test_s:
        
        row_tile = np.tile(row, (n, 1))
        distances = np.sqrt(np.sum(np.square(row_tile-X_scaled), axis=1))
        
        sorted_idxs = np.argsort(distances)
        k_sorted_idxs = sorted_idxs[:k]
        
        k_labels = Y_train[k_sorted_idxs]
        
        unique_labels, counts = np.unique(k_labels, return_counts = True)
        
        pred_label = unique_labels[np.argmax(counts)]
        
        y_preds.append(pred_label)

    
    return np.array(y_preds)