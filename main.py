import os
import random
import  pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
import numpy as np



songs = pd.read_csv("song_data.csv")
songs = songs.ffill()
features = ["acousticness",  "danceability","energy","key","liveness","loudness","audio_mode"]
target_variable = "song_popularity"
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

def prediction(params , x):  # make predicts using weights and bias
    weights, bias = params
    prediction = x @ weights + bias  # w1*x1 + w2*x2 .....
    return prediction
def mse(actual, predicted): # mean squared error
    return  np.mean((actual - predicted)**2) # our loss function
def mse_grad(actual, predicted):  #calcule our gradient
    return (predicted - actual) #calculer la derivée


def update_params(params, x, lr, grad):
    #x1 * g, x2 *g , x3*g
    w_grad = (x.T / x.shape[0]).dot(grad)  # calculate the derivative, the x.shape[0] is averaging the error across the rows
    b_grad = np.mean(grad, axis=0)   # the gradient for all our rows gets averaged
    params[0] -= w_grad * lr
    params[1] -= b_grad * lr
    return params

lr = 1e-4
epoch = 30000
def linear_regression_loop():
    params = init_params(train_x.shape[1])

    for i in range(epoch):
        predictions = prediction(params, train_x)
        grad = mse_grad(train_y, predictions)

        params = update_params(params, train_x, lr, grad)

        if i % 5000 == 0:
            predictions = prediction(params, valid_x)
            valid_loss = mse(valid_y, predictions)
            print(f"Epoch {i} loss: {valid_loss}")



    # Make predictions on the test set
    predictions_test = prediction(params, test_x)

    # Calculate MSE and MAE on the test set
    test_mse = mse(test_y, predictions_test)
    test_mae = mean_absolute_error(test_y, predictions_test)
    print(f"Test MSE: {test_mse}")
    print(f"Test MAE: {test_mae}")

def visualize_correlation(features_correlation, title):
    # Function to visualize correlation using scatter plot
    plt.scatter(features_correlation.index, features_correlation)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Correlation')
    plt.show()


numeric_features = songs.select_dtypes(include=[np.number]).columns

# Corrélation entre song popularity et les autres features
features_correlation_sp = songs[numeric_features].corr()['song_popularity']


# Corrélation entre danceability et les autres features
features_correlation_danceability = songs[numeric_features].corr()['danceability']

'''
# Create linear regression model
lr = LinearRegression()

# Fit the model
lr.fit(songs[["danceability"]], songs[target_variable])
songs.plot.scatter("danceability", "song_popularity")
plt.plot(songs["danceability"], lr.predict(songs[["song_popularity"]]))
#Afficher les corrélations
print("Corrélation avec Song Popularity:")
print(features_correlation_sp)

print("\nCorrélation avec Danceability:")
print(features_correlation_danceability)'''
# Assume you have the final weight and bias
final_weight = 2.5
final_bias = 1.0

# Generate some sample data for demonstration
x_data = np.random.rand(1000) * 10  # Random x values
y_data = final_weight * x_data + final_bias + np.random.normal(0, 1, 1000)  # Linear relation with some noise

# Plot the data points
plt.scatter(x_data, y_data, label='Data Points')

# Plot the line with the final weight and bias
x_line = np.linspace(min(x_data), max(x_data), 1000)
y_line = final_weight * x_line + final_bias
plt.plot(x_line, y_line, color='red', label='Regression Line')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Show the plot
plt.show()
if __name__ == "__main__":

    visualize_correlation(features_correlation_danceability, 'Corrélation avec Danceability')
    # Visualize correlation with song popularity
    visualize_correlation(features_correlation_sp, 'Corrélation avec Song Popularity')

    # Visualize correlation with danceability
    linear_regression_loop()






