import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from k_nearest_neighbor import knn

if __name__ == "__main__":
    
    df = pd.read_csv('data.csv')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

    print('rows in train data ',  X_train.shape[0])

    k = 5

    preds = knn(X_train, X_test, y_train, k)
    
    print('result on test data')
    print(classification_report(y_test, preds))