import random
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model






""" extracts the features associated to each song title and normalizes them using module sklearn_preprocessing 
in order to tranform the data into [0,1] values so it will be easier to calculate and make the results more balanced


	Args: 
		list_parts: a list of floats (list[float])
	returns: 
		normalized_features_list: list containing the values of each feature but fitted to data 

 """
def extract_and_normalize_features(list_parts):
	# using the MinMaxScaler from sklearn.preprocessing, it takes a 2D tab so we reshape list_parts into a 2D tab
	list_2d = [[feature] for feature in list_parts]
	scaler = MinMaxScaler()
	normalized_features = scaler.fit_transform(list_2d)  # fit to data, then transform it
	# we re reshape it again into a list
	normalized_features_list = [features[0] for features in normalized_features]
	return normalized_features_list



""" Parameters: filename (str)
	returns: songs_dict (defaultdict, key= song title, value = features), score(list) """



""" Reads data from input file and makes it readable for the model by creating a dictionary of songs with song_title as a key and a list of floats representing features as values
	Args: filename (str)
	returns: songs_dict (defaultdict, key= song title, value = features), score(list)

	 """


def read_dataset(filename):
    popularity_scores = []  # song_popularity (what we are seeking to predict), value between 0 and 100 for now ( TODO NORMALIZE THE VALUE INTO A [0,1] VALUE)

    # create a dictionnary {(song title, features)}
    songs_dict = {}

    try:
        with open(filename, 'r') as input_file:
            header = input_file.readline()
            if not header:
                print("input file is empty")
                return  # incase the file is empty

            header_parts = [header.strip() for part in header.strip().split(
                ',')]  # ['song_name', 'song_popularity', 'song_duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature', 'audio_valence']
            lines = input_file.readlines()

            for line in lines:
                line_parts = [part.strip() for part in line.strip().split(',')]
                song_name = line_parts[0]  # song's title

                # don't add it if it's already in songs_dict
                if song_name not in songs_dict:
                    if len(line_parts) == len(header_parts):  # check that the data in each line aligns with column name
                        features_list = extract_and_normalize_features(line_parts[1:])
                        score = features_list[0]
                        songs_dict[song_name] = features_list[1:]
                        popularity_scores.append(score)
                else:
                    continue
            return songs_dict, popularity_scores

    except FileNotFoundError:
        print("File does not exist: {filename}")


""" Separates the data set into two sub-sets: training_set and test_set. Pseudo-randomly
	Args: 
		input: string, name of the input file containg our dataset (file)
		seed: integer, the seed of the pseudo-random generator uses. using the same seed and same input generates the same outputs (but do we want that ?), and using different seeds generates different results 
		output1: string, name of the first output file 
		output2: string, name of the second output file 
		ratio: float between 0 and 1, to get the size of the output1 file (for example ratio= 0.65 if we want a ratio of 65% training and 35% test)

 """


def split_lines(input, seed, output1, output2, ratio):
    try:
        with open(input, 'r') as input_file:
            random.seed(seed)
            header = input_file.readline()
            if not header:
                return  # in case the file is empty

            lines = input_file.readlines()

            # counting the size of each file with the given ratio:
            input_size = len(lines)
            output1_capacity = int(
                input_size * ratio)  # Training set size, which makes the size of the test size as: input_size - training_set_size
            output2_capacity = input_size - output1_capacity
            random.shuffle(lines)  # shuffles the lines randomly

            with open(output1, 'w') as out1, open(output2, 'w') as out2:
                out1.write(header)
                out2.write(header)
                for line in lines:
                    if random.randint(0, 1) and output1_capacity > 0:
                        out1.write(line)
                        output1_capacity -= 1  # at each line written on this file, we decrease its capacity
                    elif output2_capacity > 0:
                        out2.write(line)
                        output2_capacity -= 1
                    else:
                        out1.write(line)
                        output1_capacity -= 1
    except FileNotFoundError:
        print("File does not exist : {input}")


def linear_regression_model():
    songs_dict, popularity_scores = read_dataset('train.csv')

    if not songs_dict or not popularity_scores:
        print("Error reading dataset")
        return


    # Convert the dictionary data to NumPy arrays
    X = np.array([song_values[1:] for song_values in songs_dict.values()])
    Y = np.array(popularity_scores)

    print(f"X.shape: {X.shape}")
    print(f"Y.shape: {Y.shape}")

    # Create plots for each feature
    features = ['song_duration', 'accousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
                'loudness']
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    fig.suptitle('Scatter Plots for Features vs Popularity', y=1.02)

    for i, ax in enumerate(axes.flatten()):
        ax.scatter(X[:, i], Y, marker='o', s=30, alpha=0.5)
        ax.set_title(features[i])
        ax.set_xlabel(features[i])
        ax.set_ylabel('Popularity Score')

    plt.tight_layout()
    plt.show()


"""we chose to use csv files for our data becasuse it is tabular with rows and columns representing different features """

if __name__ == "__main__":
	songs_dictio, scores= read_dataset('song_data.csv')
	split_lines('song_data.csv', 56, 'train.csv', 'test.csv', 0.65)
	linear_regression_model()




