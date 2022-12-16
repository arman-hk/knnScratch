import numpy as np
# k nearest neighbour from scratch with numpy
class knn:
    def __init__(self, n_neighbours = 5): # default for num of neighbours
        self.n_neighbours = n_neighbours
    
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
            neighbours = np.array(distances).argsort()[: self.n_neighbours]
            count = {}
            for i in neighbours:
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
        neighbours = np.array(distances).argsort()[:self.n_neighbours]
        display_val = []
        for i in range(len(neighbours)):
            neighbour_index = neighbours[i]
            e_distances = distances[i]
            display_val.append((neighbour_index, e_distances))
        return display_val
