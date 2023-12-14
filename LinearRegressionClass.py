import numpy as np


# A REFAIRE POUR LE KNN
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param = 0.01, n_iters=1000):  # lr being the learning rate and n_iters being the numberof iterations
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None  # a
        self.bias = None  # b

    def fit(self, X, Y):  # takes the training samples and the labels and it is the training step and gradient descent
        # init param
        n_samples, n_features = X.shape
        y_ = np.where(Y<= 0, -1, 1)
        #init weights
        self.weights = np.zeros(n_features) #for each compononent we put in a zero
        self.bias = 0

        # learn the weights with the update rule
        for _ in range (self.n_iters):
            for id, Xi in enumerate(X):
                predicted_y = np.dot(X, self.weights) + self.bias
                dw = (1/n_samples) * np.dot(X.T, (predicted_y - Y))
                db = (1/n_samples) * np.sum(predicted_y-Y)

            self.weights -= self.lr * dw
            self.bias -= self.lr *db


    def predict(self,X):  # takes test samples and approximates the label and return it
        pass