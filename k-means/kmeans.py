import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

def kMeans(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """
    sc = StandardScaler()
    sc.fit(X_train)

    X_train = sc.transform(X_train)
    
    clusters = {}
    
    init_centroids = np.random.randint(0, X_train.shape[0], N)
    
    centroids = X_train[init_centroids,:]
    
    ## initializing clusters
    for i,c in enumerate(centroids):
        clusters[i] = []
    
    
    
    ### check for change in clusters
    change = True
    
    ### store last clusters
    prev_clusters = np.zeros((X_train.shape[0], N))
    
    
    while change:
        
        
        ### calculate distance of data points with each cluster
        
        cluster_distances = np.zeros((X_train.shape[0], N))
        
        for i, centroid in enumerate(centroids):
            
            centroid_tile = np.tile(centroid, (len(X_train), 1))
            distances = np.sqrt(np.sum(np.square(centroid_tile-X_train), axis=1))
            cluster_distances[:,i] = distances
        
        
        
        ### assigining clusters with min distances
        new_clusters = np.argmin(cluster_distances, axis = 1)
        
        ### calculate new centroids
        new_centroids = np.zeros((N, X_train.shape[1]))
        for i in range(len(new_centroids)):
            
            i_cluster = (new_clusters == i)
            #print(i_cluster.sum())
            new_centroids[i,:] = X_train[i_cluster, :].mean(axis = 0)
            
        ### update centroids
        centroids = new_centroids.copy()
        
        ### check if cluster same as previous
        if not np.array_equal(new_clusters, prev_clusters):
            prev_clusters = new_clusters.copy()
        
        else:
            change = False
        
    
    X_clusters = []
    
    for i in range(N):
        i_cluster = (prev_clusters == i)
        X_clusters.append(X_train[i_cluster, :])
        
    
    return X_clusters

def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
    avg_distances = np.zeros(len(Clusters))
    for i, cluster in enumerate(Clusters):
        
        centroid = cluster.mean(axis = 0)
        centroid_tile = np.tile(centroid, (len(cluster), 1))
        distances = np.sqrt(np.sum(np.square(centroid_tile-cluster), axis=1))
        avg_distances[i] = distances.mean()
    
    
    return avg_distances.mean()