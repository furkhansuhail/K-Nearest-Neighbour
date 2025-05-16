import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets._samples_generator import make_blobs
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from pandas import DataFrame
from collections import Counter

class KNNClassification:
    def __init__(self):
        self.Y_train = None
        self.X_train = None
        self.ImportData()

        new_points = np.array([[-10, -10],
                               [0, 10],
                               [-15, 10],
                               [5, -2]])

        new_points = self.normalize(new_points)
        knn = self.find_neighbors(4, new_points, new_points[1])
        result = self.classifier(knn)
        print(result)
        print("_______________________________________________________________________")


    def ImportData(self):
        self.X_train, self.Y_train = make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=6, random_state=11)
        df = DataFrame(dict(x=self.X_train[:, 0], y=self.X_train[:, 1], label=self.Y_train))
        colors = {0: 'blue', 1: 'orange'}
        fig, ax = plt.subplots(figsize=(8, 8))
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
        plt.xlabel('X_1')
        plt.ylabel('X_2')
        plt.show()
        test = self.normalize(self.X_train)
        print(test[0:5])

    def normalize(self, X_train):
        X = X_train.copy()

        x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

        X[:, 0] = (X[:, 0] - x1_min) / (x1_max - x1_min)
        X[:, 1] = (X[:, 1] - x2_min) / (x2_max - x2_min)

        return X

    def find_neighbors(self, k, X_tr, new_point):
        neighbor_arr = []
        for i in range(len(X_tr)):
            dist = np.sqrt(sum(np.square(X_tr[i] - new_point)))
            neighbor_arr.append([i, dist])
        neighbor_arr = sorted(neighbor_arr, key=lambda x: x[1])

        return neighbor_arr[0:k]

    def classifier(self, neighbor_arr):
        class_arr = [self.Y_train[i[0]] for i in neighbor_arr]
        return Counter(class_arr).most_common(1)[0][0]


KNNClassificationObj = KNNClassification()

