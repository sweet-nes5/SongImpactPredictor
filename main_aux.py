import matplotlib.pyplot as plt 
import numpy as np 
import random
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler 
import seaborn as sns 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error



from sklearn.metrics import mean_squared_error
import math 

visualization = False #change value only if you want to activate the visualization analysis 
debug = False #change this value only if you want to activate the fnctions for debugging

"""shape of songs_dict:
key 'FAKE LOVE': {'song_duration': 0.1218993209862618, 'acousticness': 0.002679361088284217, 'danceability': 0.5364381198792584, 'energy': 0.7205889743203178, 
'instrumentalness': 0.0, 'key': 0.18181818181818182, 'liveness': 0.30254089085485025, 'loudness': 0.8407892676306311, 'speechiness': 0.0, 'tempo': 0.016111473982146744, 
'time_signature': 0.13577612198562228, 'audio_valence': 0.75, 'audio_mode_1': 0.3331249348754819}

""" 

def affichage_utiles(x_train, y_train, x_test, y_test): #for debug
	print(f"key 'FAKE LOVE': {x_train['FAKE LOVE']}")
	
	print(f"\nkey 'Dynamite': {x_test['Dynamite']}")
	
	print("test y : ", y_test[0])
	print("train y : ", y_train[0])

	print(len(x_train['FAKE LOVE']))

#APPLY TO THIS COLUMN TRANFORMATION!!!!

def fit_data(songs_dict):
	df = pd.DataFrame.from_dict(songs_dict, orient='index', columns= ['song_duration', 'acousticness', 'danceability', 'energy', 
			'instrumentalness', 'key', 'liveness', 'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature', 'audio_valence'])
	#df_encoded = pd.get_dummies(df, columns = categorical_features, drop_first= True)
	songs_dict_encoded = df.to_dict(orient = 'index')

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
def normalize_dataset(x_train, x_test, categorical_features= ['audio_mode']):
	#normalizing num values with MinMax Scaler 
	#num_values_train = [x_train[key] for key in x_train.keys()]
	#num_values_test = [x_test[key] for key in x_test.keys()]
	#convert dictionnaries to dataframes :
	df_train = pd.DataFrame.from_dict(x_train, orient = 'index')
	df_test = pd.DataFrame.from_dict(x_test, orient = 'index')


	num_columns = [col for col in df_train.columns if col not in categorical_features]
	num_values_train = df_train[num_columns].values 
	num_values_test = df_test[num_columns].values



	#normalize the numerical (continues values) features
	scaler = MinMaxScaler()
	normalized_num_features_train = scaler.fit_transform(num_values_train)
	normalized_num_features_test = scaler.transform(num_values_test) #using the same scaler for the test subset


	#for ordinal values like key[1, 2, ..., 11] and time_signature [0,1,2,3,4,5], we will try to convert them into numerical values
	#??? don't knwo how to do that for now and some articles say that it's not necessary to do that so we'll try and see 

	#reintegrate the new values into the dictionnaries

	for i , feature in enumerate(num_columns):
		df_train[feature] = normalized_num_features_train[:, i]

	for i, feature in enumerate(num_columns):
		df_test[feature] = normalized_num_features_test[:, i]


	return df_train.to_dict(orient = 'index'), df_test.to_dict(orient = 'index')

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

			header_parts = [part.strip() for part in header.strip().split(',')]  #  ['song_name', 'song_popularity', 'song_duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature', 'audio_valence']			
			lines = input_file.readlines()


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
					selected_output = random.choices([out1, out2], weights = [ratio, 1 - ratio])[0]

					if selected_output == out1 and output1_capacity > 0:
							out1.write(line)
							output1_capacity -= 1 #at each line written on this file, we decrease its capacity 
					elif output2_capacity > 0:
							out2.write(line)
							output2_capacity -= 1
					else:
						out1.write(line)
						output1_capacity -= 1
	except FileNotFoundError: 
		print(f"File does not exist : {input}")


"""THIS PART IS ABOUT EDA (Exploratory Data Analysis) 
about eda: its purpose is to gain information and intuition about the data; to make comparisons between distributions; for making sure the data is on the scalre we expect it to be, and in the format it should be in; 
to find where data is missing and if there are outliers, and to summarize the data. (source: from the book 'Doing Data Science by Cathy O'Neil and Rachel Schutt') 

"""

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


def remove_feature(df, feature, dictionaries):
	#remove the feature from the dataframe first 
	df = df.drop(columns = [feature])

	#update all the song_dicts
	for dictionary in dictionaries : 
		for key in dictionary:
			if feature in dictionary[key]:
				del dictionary[key][feature]



"""finds the outliers in our dataset, Creates a boxPlot from the library SeaBorn (source: python-graph-gallery and freecodecamp.com 'How to Build a Linear Regression Model – Machine Learning Example')
	
"""
def identify_outliers(x_train):
	#orient is the orientation of the data, means that the keys of the dictionnaries will be used as the index of the data frame
	df = pd.DataFrame.from_dict(x_train, orient = "index", columns = ['song_duration', 'acousticness', 'danceability', 'energy', 
			'instrumentalness', 'key', 'liveness', 'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature', 'audio_valence'])

	#they're all numerical columns 
	all_features = df.columns
	num_features =  len(all_features)
	num_cols = math.ceil(num_features / 2) #2 is the number of rows 


	
	#we create subplots so that we can compare multiple boxplots side by side, each representing a different numerical column of our DataFrame
	fig, ax = plt.subplots(2, num_cols, figsize=(15, 6), dpi= 100)

	#sns.boxplot(x = df["song_duration"])
	#display of all the columns boxplot side by side by iterating through each column (feature), we also get the indx of the column 
	for i, feature in enumerate(all_features):
		x = i // num_cols
		y = i % num_cols

		#iterate throuhh each feature and create its subplot
		sns.boxplot(x = df[feature], ax = ax[x, y])
		ax[x, y].set_xlabel(feature, size= 14)
		#ax[x, y].tick_params(axis = 'x', rotation = 45)

	plt.tight_layout()
	plt.show()

#interpretation: we can observe an asymetry between the values for the features 'acousticness', 'key' and 'liveness' 
#on peut corriger les outliers, qui ont des valeurs aberrantes (trop élévés ou trop basses) et les rendre moins influentes
#on peut les remplacer par les medianes ou bien en appliquant une transformation logarithmique dessus 
def correct_outliers(x_train, feature): #A CORRIGER 
	df = pd.DataFrame.from_dict(x_train, orient = "index", columns = ['song_duration', 'acousticness', 'danceability', 'energy', 
			 'key', 'liveness', 'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature', 'audio_valence'])
	
	df[feature] = np.log1p(df[feature])
	for key, values in df.to_dict(orient='index').items():
		x_train[key][feature] = values[feature]
	return x_train



########################################################################################################################################
#Model Training 
#we'll use a multi linear regression model but first we have to create a linear regression model 

def fit_linear_regression(x, y, learning_rate = 0.00001, epsilon= 0.9):
	"""Fit a linear regression using gradient descent
	Args: 
		x: must be songs_dict in my case songs_dict= { 'song_itle': {'feature1': value, 'feature2': value, .....}, 'song_title': {...}, ...}
		y: must be the popularity scores in my case 
		leanring_rate: tje learning rate factor number 
		epsilon: float, the error threshold, when the error is lesser than the epsilon then the model converged

	returns: 
		nb.array: Array wih the regression weights 

	"""


	#Step 1 : Insert a new column with ones for y-intercept 
	regression = np.c_[x, np.ones(len(x))]

	#Step 2: Declare the weights with the same width than x 
	weights = np.ones(regression.shape[1])

	#Step 3: Implement gradient descent 
	norma = 1 
	while (norma > epsilon ):
		#Step 3.1 compute the partial 
		y_pref = regression @ weights.T 
		partiel = regression.T @ (y - y_pred)
		#step 3.2: compute the norma
		norma = np.sum(np.sqrt(np.quare(partial)))

		#step 3.3 ajust the weights 
		weights = weights.T + (learning_rate * partial)
		if(np.isnan(norma)):
			warnings.warn('the model diverged, try to use smaller learning rate')
	return weights 





############################################################################################"""
def linear_regression_loop():
    params = init_params(train_x.shape[1])

    for i in range(epoch):
        predictions = forward(params, train_x)
        grad = mse_grad(train_y, predictions)

        params = backward(params, train_x, lr, grad)

        if i % 5000 == 0:
            predictions = forward(params, valid_x)
            valid_loss = mse(valid_y, predictions)
            #print(f"Epoch {i} loss: {valid_loss}")



    # Make predictions on the test set
    predictions_test = forward(params, test_x)

    # Calculate MSE and MAE on the test set
    test_mse = mse(test_y, predictions_test)
    test_mae = mean_absolute_error(test_y, predictions_test)
    print(f"Test MSE: {test_mse}")
    print(f"Test MAE: {test_mae}")

def mse(actual, predicted): # mean squared error
    return  np.mean((actual - predicted)**2)
def mse_grad(actual, predicted):  #calcule our gradient
    return (predicted - actual)

def init_params(features):  # initialize weights and the bias for our predictors(features)
    np.random.seed(3)
    weights = np.random.rand(features, 1) # nfeature weights between 0 and 1
    biases = np.ones((1, 1))
    return [weights, biases]

def forward(params , x):  # make predicts using weights and bias
    weights, bias = params
    prediction = x @ weights + bias  # w1*x1 = w2*x2 .....
    return prediction

#def cross_validation(model, X, y, n_splits=5, random_state=42):


def linear_regression_loop(train_x, train_y, valid_x, valid_y, test_x, test_y, epoch = 10000, lr= 0.001):
	 # Initialize parameters
    params = init_params(train_x.shape[1])

    for i in range(epoch):
        # Forward pass
        predictions = forward(params, train_x)
        
        # Compute gradient
        grad = mse_grad(train_y, predictions)

        # Backward pass
        params = backward(params, train_x, lr, grad)

        if i % 5000 == 0:
            # Evaluate on the validation set
            predictions = forward(params, valid_x)
            valid_loss = mean_squared_error(valid_y, predictions)
            print(f"Epoch {i} loss: {valid_loss}")

    # Make predictions on the test set
    predictions_test = forward(params, test_x)

    # Calculate MSE and MAE on the test set
    test_mse = mean_squared_error(test_y, predictions_test)
    test_mae = mean_absolute_error(test_y, predictions_test)
    print(f"Test MSE: {test_mse}")
    print(f"Test MAE: {test_mae}")

######################################################################################

def linear_regression_loop_with_cross_validation(X, y, epoch=10000, lr=0.001, n_splits=5, random_state=42):
	kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
	test_mse_scores = []
	test_mae_scores = []

	X = np.array(X)  # Convert X to a NumPy array if it's not already
	y = np.array(y)  # Convert y to a NumPy array if it's not already


	for fold, (train_index, test_index) in enumerate(kf.split(X)):
		X_train, X_valid = X[train_index], X[test_index]
		y_train, y_valid = y[train_index], y[test_index]

		# Initialize parameters*
		params = init_params(len(X_train[0]))

		for i in range(epoch):

		# Forward pass
			predictions_train = forward(params, X_train)
			
			# Compute gradient
			grad_train = mse_grad(y_train, predictions_train)
			# Backward pass

			params = backward(params, X_train, lr, grad_train)

			if i % 5000 == 0:
			# Evaluate on the validation set

				predictions_valid = forward(params, X_valid)
				valid_loss = mean_squared_error(y_valid, predictions_valid)

				print(f"Fold {fold + 1}, Epoch {i} validation loss: {valid_loss}")

		# Make predictions on the test se

		predictions_test = forward(params, X[test_index])

		# Calculate MSE and MAE on the test set

		test_mse = mean_squared_error(y[test_index], predictions_test)
		test_mae = mean_absolute_error(y[test_index], predictions_test)

		test_mse_scores.append(test_mse)
		test_mae_scores.append(test_mae)

		print(f"\nFold {fold + 1} Test MSE: {test_mse}")

		print(f"Fold {fold + 1} Test MAE: {test_mae}")

	# Print average scores
	print("\nAverage Scores:")
	print(f"Average Test MSE: {np.mean(test_mse_scores)}")
	print(f"Average Test MAE: {np.mean(test_mae_scores)}")	






























"""we chose to use csv files for our data becasuse it is tabular with rows and columns representing different features """

if __name__ == "__main__":
	split_lines('song_data.csv', 56, 'train.csv', 'test.csv', 0.8)
	songs_dict_train, scores_train = read_dataset('train.csv')

	songs_dict_test, scores_test = read_dataset('test.csv')

	songs_dict_train, songs_dict_test = normalize_dataset(songs_dict_train, songs_dict_test)
	
	df = pd.DataFrame.from_dict(songs_dict_train, orient= 'index', columns = ['song_duration', 'acousticness', 'danceability', 'energy', 
			'instrumentalness', 'key', 'liveness', 'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature', 'audio_valence'])

	



	if visualization == True: 
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

		identify_outliers(songs_dict_train)

		songs_dict_train_log = correct_outliers(songs_dict_train, 'liveness')
		identify_outliers(songs_dict_train_log)

	#the visuals show that instrumentalness has a lot of values at 0.0 and does not influence song_popularity
	remove_feature(df, 'instrumentalness', [songs_dict_train, songs_dict_test])
	if debug == True: 
		affichage_utiles(songs_dict_train, scores_train, songs_dict_test, scores_test)
	#accousticness also have a lot of null values, should we remove it too ? 
	#let's analyse its correlation with the target feature: 


	
	song_features = np.array([list(song.values()) for song in songs_dict_train.values()], dtype = np.float64)
	scores_train_array = np.array(scores_train, dtype = np.float64)
	scores_test_array = np.array(scores_test, dtype = np.float64)

	linreg = LinearRegression()
	linreg.fit(song_features ,scores_train_array)

	songs_test = np.array([list(song.values()) for song in songs_dict_test.values()], dtype = np.float64)
	y_pred = linreg.predict(songs_test)

	accuracy = r2_score(scores_test_array, y_pred)

	print(accuracy)



#decision tree 


	# Create a DecisionTreeRegressor
	tree_reg = DecisionTreeRegressor()

	# Fit the model on the training data
	tree_reg.fit(song_features, scores_train_array)

	# Make predictions on the test data
	predictions = tree_reg.predict(songs_test)

	# Evaluate the model using Mean Squared Error (MSE)
	mse = mean_squared_error(scores_test_array, predictions)
	print(f"Mean Squared Error: {mse}")
		

	accuracy = r2_score(scores_test_array, predictions)

	print(accuracy)










# Example usage
	#linear_regression_loop_with_cross_validation(songs_dict_train, scores_train)





