import os
import random
import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
import numpy as np
directory = './'



songs = pd.read_csv("song_data.csv")
songs = songs.ffill()

features = ["acousticness", "song_popularity","energy","key","liveness","loudness","audio_mode"]
target_variable =  "danceability"
songs = songs.dropna(subset=features)
songs = songs.drop_duplicates(subset=features)
np.random.seed(0)
split_data = np.split(songs, [int(.7 * len(songs)), int(.85 * len(songs))])
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = [[d[features].to_numpy(), d[[target_variable]].to_numpy()] for d in split_data]

def init_params(features):  # initialize weights and the bias for our predictors(features)
    np.random.seed(3)
    weights = np.random.rand(features, 1) # nfeature weights between 0 and 1
    biases = np.ones((1, 1))
    return [weights, biases]

def forward(params , x):  # make predicts using weights and bias
    weights, bias = params
    prediction = x @ weights + bias  # w1*x1 = w2*x2 .....
    return prediction
def mse(actual, predicted): # mean squared error
    return  np.mean((actual - predicted)**2)
def mse_grad(actual, predicted):  #calcule our gradient
    return (predicted - actual)


def backward(params, x, lr, grad):
    #x1 * g, x2 *g , x3*g
    w_grad = (x.T / x.shape[0]) @ grad  # calculate the derivative, the x.shape[0] is averaging the error across the rows
    b_grad = np.mean(grad, axis=0)   # the gradient for all our rows gets averaged
    params[0] -= w_grad * lr
    params[1] -= b_grad * lr
    return params

lr = 1e-4
epoch = 50000
def linear_regression_loop():
    params = init_params(train_x.shape[1])

    for i in range(epoch):
        predictions = forward(params, train_x)
        grad = mse_grad(train_y, predictions)

        params = backward(params, train_x, lr, grad)

        if i% 5000 == 0:
            predictions = forward(params, valid_x)
            valid_loss = mse(valid_y, predictions)
            print(f"Epoch {i} loss: {valid_loss}")


if __name__ == "__main__":
    linear_regression_loop()