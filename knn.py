import numpy as np
# k nearest neighbor from scratch with numpy
class knn:
    def __init__(self, n_neighbors = 5): # default for num of neighbors
        self.n_neighbors = n_neighbors
    
    # euclidian distance func
    def euc_dis(self, p, q):
        dis = 0
        for i in range(len(q)):
            dis += (p[i] - q[i]) ** 2
            dis = np.sqrt(dis)
        return dis
    # fitting func
    def fit(self, X, y):
        self.X = X
        self.y = y
    # predict func
    def predict(self, X):
        pred = []
        for i in range(len(X)):
            for i in self.X:
                distances = []
                distance = self.euc_dis(i, X[i])
                distances.append(distance)
            neighbors = np.array(distances).argsort()[: self.n_neighbors]
            count = {}
            for i in neighbors:
                if self.y[i] in count:
                    count[self.y[i]] += 1
                else:
                    count[self.y[i]] = 1
            pred.append(max(count, key=count.get))
        return 
    # display knn func 
    def display(self, a):
        distances = []
        for i in self.X:
            distance = self.euc_dis(i, a)
            distances.append(distance)
        neighbors = np.array(distances).argsort()[:self.n_neighbors]
        display_val = []
        for i in range(len(neighbors)):
            neighbor_index = neighbors[i]
            e_distances = distances[i]
            display_val.append((neighbor_index, e_distances))
        return display_val
