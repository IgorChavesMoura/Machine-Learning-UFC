import numpy as np


def squared_euclidean_distance(xi, xj):
    return np.sum((xi - xj)**2)

def batch_squared_euclidean_distance(x1, x2):
    return -2 * x1 @ x2.T + np.sum(x1**2, axis=1)[:,None] + np.sum(x2**2, axis=1)

def manhattan_distance(xi, xj):
    return np.sum(np.abs(xi - xj))

def batch_manhattan_distance(x1, x2):
    return np.abs(x1[:,None] - x2).sum(-1)

def squared_mahalanobis_distance(xi, xj, cov_matrix, inv_cov_matrix=None):
    if inv_cov_matrix is None:
        inv_cov_matrix = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * np.mean(np.diag(cov_matrix)) * 10 ** -6)
        
    return np.sum((xi - xj).T @ inv_cov_matrix @ (xi - xj))

def batch_squared_mahalanobis_distance(x1, x2, cov_matrix):
    inv_cov_matrix = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * np.mean(np.diag(cov_matrix)) * 10 ** -6)    
        
    return np.sum((x1 @ inv_cov_matrix) * x1, axis=1)[:,None] + \
           np.sum((x2 @ inv_cov_matrix) * x2, axis=1) - 2 * (x1 @ inv_cov_matrix @ x2.T)

def create_batch_distance_matrix(x_new, x, distance_metric='euclidean'):
    
    distance_metric = distance_metric.lower()
    
    if distance_metric == 'manhattan':
        dist_matrix = batch_manhattan_distance(x_new, x)
        
    elif distance_metric == 'mahalanobis':
        mu = np.mean(x, axis=0)
        cov_matrix = (x - mu).T @ (x - mu) / x.shape[0]        
        dist_matrix = batch_squared_mahalanobis_distance(x_new, x, cov_matrix=cov_matrix)
        
    else:
        dist_matrix = batch_squared_euclidean_distance(x_new, x)
    
    return dist_matrix

def reconstrucion_error(x,partitions,centroids):
    
    rec_error = 0
    
    for centroid,partition in zip(centroids,partitions):
        
        for i in partition:
            
            rec_error += np.sum((x[i] - centroid)**2)
    
    
    return rec_error
    
    
def create_partitions(x,centroids):
    
    partitions = [[] for k in range(centroids.shape[0])]
    
#    dist_matrix = create_batch_distance_matrix(centroids,x)
    
    for i in range(x.shape[0]):
        
        selected_centroid = -1
        curr_dist = -1
        
        for k in range(centroids.shape[0]):
        
            
            if selected_centroid == -1 or dist < curr_dist:
            
                selected_centroid = k
                curr_dist = dist
        
        partitions[selected_centroid].append(i)
      
    
    return partitions
    
def kmeans(x, K, num_rep):
    
    
    loss_history = []
    
    while True:
    
        partitions = create_partitions(x,centroids)

        new_centroids = []

        for centroid,partition in zip(centroids,partitions):

            new_centroid = np.array(x[partition]).sum(axis=0)
            new_centroid /= len(partition)

            new_centroids.append(new_centroid)
            
        
        loss_history.append(reconstruction_error(partitions,centroids))
        
        if np.array_equal(centroids,new_centroids):
            
            break
            
        centroids = new_centroids
        
    
    return {
        
        'cluster_index':partitions,
        'centroids':centroids,
        'loss_history':loss_history
        
    }
    
