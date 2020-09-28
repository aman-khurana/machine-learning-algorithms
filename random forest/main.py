import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from random_forest import build_forest,  predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_and_predict(X_train,Y_train,X_test):
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
    
    n_trees = 3
    print('number of trees to be built in forest ', n_trees)
    trees = build_forest(X_scaled, Y_train, n_trees, 
                                            min_sample_split = 12,
                                            max_depth = 10, 
                                            min_gain_threshold = 0.001, 
                                            feature_subset = False)
    
    rf_pred = predict(X_test_s, trees)
    
    return rf_pred


if __name__ == "__main__":
    
    ## data file with last column as labels and all non categorical columns
    df = pd.read_csv('data.csv')
    
    X = df.values[:,:-1]
    y = df.values[:,-1] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

    print('rows in train data ',  X_train.shape[0])
    
    preds =  train_and_predict(X_train, y_train, X_test)

    print('result on test data')
    print(classification_report(y_test, preds))