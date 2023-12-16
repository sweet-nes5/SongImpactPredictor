import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def extract_and_normalize_features(list_parts, categorical_indx=[]):
    num_variables = [list_parts[i] for i in range(len(list_parts)) if i not in categorical_indx]

    list_2d = [[feature] for feature in num_variables]
    scaler = MinMaxScaler()
    normalized_num_features = scaler.fit_transform(list_2d)

    normalized_num_features_list = [features[0] for features in normalized_num_features]

    categorical_features = [list_parts[i] for i in categorical_indx]

    normalized_features = []
    num_list_indx = 0

    for i in range(len(list_parts)):
        if i in categorical_indx:
            normalized_features.append(list_parts[i])
        else:
            normalized_features.append(normalized_num_features_list[num_list_indx])
            num_list_indx += 1

    return normalized_features


def read_dataset(filename):
    popularity_scores = []
    songs_dict = {}

    try:
        with open(filename, 'r') as input_file:
            header = input_file.readline()
            if not header:
                print("Input file is empty")
                return None, None

            lines = input_file.readlines()

            for line in lines:
                line_parts = [part.strip() for part in line.strip().split(',')]
                song_name = line_parts[0]

                if song_name not in songs_dict:
                    if len(line_parts) == len(header.split(',')):
                        features_list = extract_and_normalize_features(
                            line_parts[1:])
                        score = features_list[0]
                        songs_dict[song_name] = features_list[1:]
                        popularity_scores.append(score)
                else:
                    continue

        return songs_dict, popularity_scores

    except FileNotFoundError:
        print(f"File does not exist: {filename}")
        return None, None

def split_lines(input, seed, output1, output2, ratio):
    try:
        with open(input, 'r') as input_file:
            random.seed(seed)
            header = input_file.readline()
            if not header:
                return

            lines = input_file.readlines()

            input_size = len(lines)
            output1_capacity = int(input_size * ratio)
            output2_capacity = input_size - output1_capacity
            random.shuffle(lines)

            with open(output1, 'w') as out1, open(output2, 'w') as out2:
                out1.write(header)
                out2.write(header)
                for line in lines:
                    if random.randint(0, 1) and output1_capacity > 0:
                        out1.write(line)
                        output1_capacity -= 1
                    elif output2_capacity > 0:
                        out2.write(line)
                        output2_capacity -= 1
                    else:
                        out1.write(line)
                        output1_capacity -= 1

    except FileNotFoundError:
        print(f"File does not exist: {input}")

def init_params(features):  # initialize weights and the bias for our predictors(features)
    np.random.seed(3)
    weights = np.random.rand(features, 1) # nfeature weights between 0 and 1
    bias = np.ones((1, 1))
    return [weights, bias]

def forward(params , x):  # make predicts using weights and bias
    weights, bias = params
    prediction = x @ weights + bias  # w1*x1 = w2*x2 .....
    return prediction

def loss_function(actual, predicted): # mean squared error
    return  np.mean((actual - predicted)**2)

def loss_grad(actual, predicted):  #calcule our gradient
    return (predicted - actual)

def backward(params, x, lr, grad):
    #x1 * g, x2 *g , x3*g
    w_grad = (x.T / x.shape[0]) @ grad  # calculate the derivative, the x.shape[0] is averaging the error across the rows
    b_grad = np.mean(grad, axis=0)   # the gradient for all our rows gets averaged
    print("Shape of params[0]:", params[0].shape)
    print("Shape of w_grad:", w_grad.shape)
    params[0] -= w_grad * lr
    params[1] -= b_grad * lr
    return params

def split_train_val():
    # Separate the training data to training and validation
    split_lines('train.csv', 52, 'train.csv', 'validation.csv', 0.65)

    # Read training data
    train_songs_dict, train_popularity_scores = read_dataset('train.csv')

    # Read validation data
    val_songs_dict, val_popularity_scores = read_dataset('validation.csv')

    # Check if the datasets are loaded successfully
    if train_songs_dict is not None and val_songs_dict is not None:
        # Extract features from dictionaries
        X_train = np.array([list(train_songs_dict[song_name]) for song_name in train_songs_dict])
        Y_train = np.array(train_popularity_scores)

        X_val = np.array([list(val_songs_dict[song_name]) for song_name in val_songs_dict])
        Y_val = np.array(val_popularity_scores)

        return (X_train, Y_train), (X_val, Y_val)
    else:
        return None, None

(train_data, val_data) = split_train_val()
if train_data is not None:
    # Extract X_train and Y_train from the tuple
    X_train, Y_train = train_data
if val_data is not None:
    # Extract X_train and Y_train from the tuple
    X_val, Y_val = val_data

lr = 1e-4
epoch = 100 # each time we pass the data into the algorithm it's called an epoch
def linear_regression_loop(): # until the error gets low enough, running for a 100 epochs
    params = init_params(X_train.shape[1])
    for i in range(epoch):
        predictons = forward(params, X_train)
        grad = loss_grad(Y_train, predictons)
        params = backward(params, X_train, lr, grad)

        if i % 10 == 0:
            predictions = forward(params, X_val)
            valid_loss =loss_function(Y_val , predictions)
            print(f"Epock{i} loss : {valid_loss}")

def target_visualization(x_label, df, kind='hist', bins=30):
    plt.title(f"Distribution of {x_label} frequency")
    sns.set(style="darkgrid")

    df = df.sort_values(by=x_label)
    if kind == 'hist':
        sns.histplot(data=df, x=x_label, kde=True, bins=bins)
    elif kind == 'bar':
        sns.countplot(data=df, x=x_label)

    plt.ylabel('Frequency')
    plt.show()

def all_features_visualization(df, bins=50):
    plt.figure(figsize=(12, 8))
    sns.set(style="darkgrid")

    for i, column in enumerate(df.columns):
        plt.subplot(3, 5, i + 1)
        sns.histplot(data=df, x=column, kde=True, bins=bins)
        plt.ylabel('Frequency')
        plt.xlabel(column)
    plt.tight_layout()
    plt.show()


'''
1.Scale the data ...done to make sure that no column will dominate the others in the cluster
2. init random centroids
3.label each data point 
4. update centroids 
5. repeat 3 and 4 until centroids stop changing --> converginf'''

songs = pd.read_csv("song_data.csv")
features = ["acousticness","danceability","energy","instrumentalness","key","liveness","loudness","audio_mode"]
songs = songs.dropna(subset=features)
songs = songs.drop_duplicates(subset=features)
songs = ((songs - songs.min()) / (songs.max() - songs.min())) * 9 + 1
print(songs)



if __name__ == "__main__":
    split_lines('song_data.csv', 56, 'train.csv', 'test.csv', 0.75)
    songs_dict, scores = read_dataset('train.csv')

    x_label_popularity = "song_popularity"
    df_popularity = pd.DataFrame({x_label_popularity: scores})
    target_visualization(x_label_popularity, df_popularity, bins=50)

    #df = pd.DataFrame.from_dict(songs_dict, orient='index', columns=['song_duration', 'acousticness', 'danceability', 'energy','instrumentalness', 'key', 'liveness', 'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature', 'audio_valence'])

    x_label_feature = 'acousticness'
    #target_visualization(x_label_feature, df, kind='hist', bins=60)

    #all_features_visualization(df, bins=10)
    #linear_regression_loop()

