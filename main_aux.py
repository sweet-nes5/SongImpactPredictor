import random
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np 
from sklearn.metrics import mean_squared_error


visualization = True #change value only if you want to activate the visualization analysis 
debug = False #change this value only if you want to activate the fnctions for debugging



def affichage_utiles(x_train, y_train, x_test, y_test): #for debug
	print(f"key 'FAKE LOVE': {x_train['FAKE LOVE']}")
	
	print(f"\nkey 'Dynamite': {x_test['Dynamite']}")
	
	print("test y : ", y_test[0])
	print("train y : ", y_train[0])

	print(len(x_train['FAKE LOVE']))


def one_hot_encode(songs_dict, categorical_features):
	df = pd.DataFrame.from_dict(songs_dict, orient='index', columns= ['song_duration', 'acousticness', 'danceability', 'energy', 
			'instrumentalness', 'key', 'liveness', 'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature', 'audio_valence'])

	df_encoded = pd.get_dummies(df, columns = categorical_features, drop_first= True)
	songs_dict_encoded = df_encoded.to_dict(orient = 'index')

	return songs_dict_encoded


"""normalized the values of each feature of each song contained in songs_dict, using the MinMaxScaler() 
	and uses one-hot encoding for values tthat are categorical (0 or 1) 


	(sources: from scikit-learn.org and medium.com 'Normalize data before or after split of training and testing data?')
	
	X_normalized = (X - X_min) / (X_max - X_min), normalizes the values between 0 and 1 

	Args:
		x_train: dictionary {song_title: list of features values} of the TRAINING set
		x_test: dictionary {song_title: list of features values} of the TEST set
		
	returns:
	Both dictionaries but with normalized values 
"""
def normalize_dataset(x_train, x_test):
	#normalizing num values with MinMax Scaler 
	num_values_train = [x_train[key] for key in x_train.keys()]
	num_values_test = [x_test[key] for key in x_test.keys()]

	scaler = MinMaxScaler()
	normalized_num_features_train = scaler.fit_transform(num_values_train)
	normalized_num_features_test = scaler.fit_transform(num_values_test)


	#for categorical_features, we'll use the one-hot encoding function (features that have only 1 or 0 values, here it's audio_mode)
	#transformation too 
	categorical_features = ['audio_mode']
	x_train= one_hot_encode(x_train, categorical_features)
	x_test = one_hot_encode(x_test, categorical_features)

	#for ordinal values like key[1, 2, ..., 11] and time_signature [0,1,2,3,4,5], we will try to convert them into numerical values
	#??? don't knwo how to do that for now and some articles say that it's not necessary to do that so we'll try and see 


	#reintegrate the new values into the dictionnaries

	for indx, key in enumerate(x_train.keys()):
		for indx_feature, feature in enumerate(x_train[key].keys()):

			if feature not in categorical_features:
				x_train[key][feature] = normalized_num_features_train[indx][indx_feature]
			else:
				continue #will keep the hot one encoded value

	for indx, key in enumerate(x_test.keys()):
		for indx_feature, feature in enumerate(x_test[key].keys()):

			if feature not in categorical_features:
				x_test[key][feature] = normalized_num_features_test[indx][indx_feature]
			else:
				continue #will keep the hot one encoded value
		


	#normalizing the target variable (here popularity score) should not be necessary and would actually add difficulties when interpreting the resulsts at the end 
	#Important only if we use a Neural Network cause they're sensitive to scales
	return x_train, x_test
		

"""Reads data from input file and makes it readable for the model by creating a dictionary of songs with song_title as a key and a list of floats representing features as values
	Args: filename (str)
	returns: songs_dict (defaultdict, key= song title, value = features), score(list)"""

def read_dataset(filename):
	popularity_scores = []  # song_popularity (what we are seeking to predict), value between 0 and 100 for now ( TODO NORMALIZE THE VALUE INTO A [0,1] VALUE)

	#create a dictionnary {(song title, features)}
	songs_dict = defaultdict() 

	try:
		with open(filename, 'r') as input_file: 
			header = input_file.readline()
			if not header:
				print("input file is empty")
				return  #incase the file is empty

			header_parts = [header.strip() for part in header.strip().split(',')]  #  ['song_name', 'song_popularity', 'song_duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature', 'audio_valence']			
			lines = input_file.readlines()
			categorical_features_indeces = [0, 1, 6, 9, 12]	

			for line in lines:
				line_parts = [part.strip() for part in line.strip().split(',')]
				song_name = line_parts[0] #song's title

				#don't add it if it's already in songs_dict
				if song_name not in songs_dict:
					if len(line_parts) == len(header_parts): #check that the data in each line aligns with column name
						
						line_parts_float = [float(element) for element in line_parts[1:]]
						#features_list = extract_and_normalize_features(line_parts_float, categorical_features_indeces) #normalized features
						features_list = line_parts[1:]
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
				return #in case the file is empty

			lines = input_file.readlines()

			#counting the size of each file with the given ratio: 
			input_size= len(lines)
			output1_capacity = int(input_size * ratio)  #Training set size, which makes the size of the test size as: input_size - training_set_size 			
			output2_capacity = input_size - output1_capacity
			random.shuffle(lines) #shuffles the lines randomly 
			
			with open(output1, 'w') as out1, open(output2, 'w') as out2:
				out1.write(header)
				out2.write(header)
				for line in lines:
					if random.randint(0, 1) and output1_capacity > 0:
							out1.write(line)
							output1_capacity -= 1 #at each line written on this file, we decrease its capacity 
					elif output2_capacity > 0:
							out2.write(line)
							output2_capacity -= 1
					else:
						out1.write(line)
						output1_capacity -= 1
	except FileNotFoundError: 
		print("File does not exist : {input}")




#data analysis and visualization in order to gain a deeper understanding of our dataset; AND to identify the most important features 

#step1: understand how the target variable (song_popularity) is distributed to see if the scores are distributed evenly or is there some inbalance in the distribution. 
#If the latter is the case, then we have to see if there is outliers (points that are too different from the majority, unusually high or low score)

"""Creates a figure, histogram , or actives an existing one to display the distribution of the model's target variable (in this case it's the popylarity scores)
(sources: matplot.pyplot.figure, matplotlib.org, python-graph-gallery.com, seaborn.pydata.org )
   We will be using SeaBorn to plot it using the histplot function. 
	Args: 
		feature: list of floats, the values of the feature that we want to analyse 	
		x_label: string, label of x - axis 
 """
def target_visualization(x_label, df, kind = 'hist', bins = None):
	plt.title(f"Distribution of {x_label} frequency")
	sns.set(style = "darkgrid")

	df = df.sort_values(by=x_label)
	if kind == 'hist':
		#sorting the df by the feature column
		sns.histplot(data = df, x= x_label, kde = True, bins= bins) #if kde (Kernel Density Estimate) true, it computes a kernel density estimate to SMOOTH the distribution, if false, it will show the histogram in its raw form 
	elif kind == 'bar': #same as hist but better for categorical variables and without the curve
		sns.countplot(data= df, x= x_label)
	#elif kind == 'dist':
	#	sns.distplot(df[x_label], kde = True, bins = bins, hist_kws = dict(edgecolor = "white", linewidth = 2))
	
	plt.ylabel('Frequency')
	plt.show()


#to show all histograms for each feature on the same figure, use this function  (EXCEPT SONG_POPULARITY since it's not contained in song_dict)
#interpretation: we can see that the instrumentalness feature does not tell us a lot and has a lot of values at 0.0 which could influence badly our model
def all_features_visualization(df, bins = 50):
	plt.figure(figsize = (12,8))
	sns.set(style = "darkgrid")

	for i, column in enumerate(df.columns):
		plt.subplot(3,5, i+1) #3 rows, 5 colums (14 features)
		sns.histplot(data= df, x= column, kde= True, bins = bins)
		plt.ylabel('Frequency')
		plt.xlabel(column)
	plt.tight_layout()
	plt.show()


#source for this (medium.com, 'How to create a Seaborn Correlation Heatmap in Python')

#analyze the correlation between different features, just creates a correlation matrix of all pairs of features 
"""how to interpret the results: """
def correlation_matrix(df):
	correlation_matrix = df.corr() #pandas.DataFrame.corr: Computes pairwise correlation of columns, excluding NA/null values (from pandas.pydata.org)
	return correlation_matrix


#creates a general heatmap of all the features' correlation for better visuals using seaborn again 
def visualize_correlation(correlation_matrix):
	sns.heatmap(correlation_matrix, annot = True)  #(from seaborn.pydata.org and python-graph-gallery.com)
	plt.title('Features Correlation Heatmap')
	plt.show()


#creates a heatmap to show the correlation between the scores of popularity and the other features to see which ones are the most influent
#interpretation: we can see that instrumentalness has the lowest correlation with song_popularity
def visualize_correlation_with_popularity(df, popularity_scores):
	#adding the scores to the df since they're separated outside this function 
	df['song_popularity'] = popularity_scores
	correlation_matrix = df.corr()

	#creating the heatmap 
	sns.heatmap(correlation_matrix[['song_popularity']], annot = True, vmin= -1, vmax= 1) #VMIN 
	plt.title('Correlation between Song Popularity and Other Features')
	plt.show()



#Model Training, here x_train must be normalized 





"""we chose to use csv files for our data becasuse it is tabular with rows and columns representing different features """

if __name__ == "__main__":
	split_lines('song_data.csv', 56, 'train.csv', 'test.csv', 0.65)
	songs_dict_train, scores_train = read_dataset('train.csv')
	limit = 0

	songs_dict_test, scores_test = read_dataset('test.csv')

	songs_dict_train, songs_dict_test = normalize_dataset(songs_dict_train, songs_dict_test)
	
	#using one hot encoding for categorical variables
	
	if visualization == True: 
		df = pd.DataFrame.from_dict(songs_dict_train, orient= 'index', columns = ['song_duration', 'acousticness', 'danceability', 'energy', 
			'instrumentalness', 'key', 'liveness', 'loudness', 'audio_mode_1', 'speechiness', 'tempo', 'time_signature', 'audio_valence'])

		"""analyzing the distribution of the song_popularity variable contained in the list scores
		Interpretation: 
			- the curve has a single peak and is very smooth; it indicates a homogenous distribution """
		x_label_popularity = "song_popularity"
		df_popularity = pd.DataFrame({x_label_popularity: scores_train})

		target_visualization(x_label_popularity, df_popularity, bins= 30)


		#now we want to visualize the distribution by other features 
		#x_label_feature = 'acousticness' #example, change this depending on which feature you want to visualize
		#target_visualization(x_label_feature, df, kind = 'hist', bins =30) #adjust bins depending on each feature 


		all_features_visualization(df, bins = 10)
		corr_matrix = correlation_matrix(df)
		visualize_correlation(corr_matrix)

		visualize_correlation_with_popularity(df, scores_train)

	if debug == True: 
		affichage_utiles(songs_dict_train, scores_train, songs_dict_test, scores_test)