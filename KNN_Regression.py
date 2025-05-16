import numpy as np
from sklearn.datasets import make_regression
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


class KNNRegression:
    def __init__(self):
        self.Y_train = None
        self.X_train = None
        self.ImportData()

        new_points = np.array([[-1, 1],
                               [0, 2],
                               [-3, -2],
                               [3, -3]])

        print("Predictions for new points:")
        for point in new_points:
            knn = self.find_neighbors(3, self.X_train, point)
            result = self.regressor(knn)
            print(f"Point: {point}, Prediction: {result:.2f}")

        self.plot_predictions(new_points)

    def ImportData(self):
        self.X_train, self.Y_train = make_regression(
            n_samples=300, n_features=2, n_informative=2, noise=5, bias=30, random_state=200
        )
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.X_train[:, 0], self.X_train[:, 1], self.Y_train, c="red", alpha=0.5, marker='o', label="Training Data")
        ax.set_xlabel('X_1')
        ax.set_ylabel('X_2')
        ax.set_zlabel('Y')
        ax.legend()
        plt.title("Training Data")
        plt.show()

    def find_neighbors(self, k, X_train, new_point):
        distances = []
        for i in range(len(X_train)):
            dist = np.sqrt(np.sum((X_train[i] - new_point) ** 2))  # Euclidean distance
            distances.append((i, dist))
        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def regressor(self, neighbor_arr):
        y_values = [self.Y_train[i[0]] for i in neighbor_arr]
        return np.mean(y_values)

    def plot_predictions(self, new_points):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot training data
        ax.scatter(self.X_train[:, 0], self.X_train[:, 1], self.Y_train, c="red", alpha=0.4, marker='o', label="Training Data")

        # Predict and plot each new point
        for point in new_points:
            knn = self.find_neighbors(3, self.X_train, point)
            prediction = self.regressor(knn)
            ax.scatter(point[0], point[1], prediction, c="blue", marker='^', s=100, label="Predicted Point")

        ax.set_xlabel('X_1')
        ax.set_ylabel('X_2')
        ax.set_zlabel('Predicted Y')
        plt.title("KNN Regression Predictions")
        plt.legend(loc="best")
        plt.show()


# Instantiate and run
KNNRegressionObj = KNNRegression()
