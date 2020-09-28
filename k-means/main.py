import numpy as np
import pandas as pd

from kmeans import kMeans, WCSS


if __name__ == "__main__":
    
    df = pd.read_csv('data.csv')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    num_clusters = 14
    clusters = kMeans(X, num_clusters)
    wcss = WCSS(clusters)

    print("Within Cluster Sum of Squares score {} s".format(round(wcss, 2)))
    