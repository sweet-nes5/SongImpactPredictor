import random
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd


""" extracts the features associated to each song title and normalizes them using module sklearn_preprocessing 
in order to tranform the data into [0,1] values so it will be easier to calculate and make the results more balanced


	Args: 
		list_parts: a list of floats (list[float])
	returns: 
		normalized_features_list: list containing the values of each feature but fitted to data 

 """
def extract_and_normalize_features(list_parts, categorical_indx=[]): 
	#separate the numerical (continuous) and categorical (discreet) variables cause they won't be treated the same, 
	#we exclude the categorical features from normalization  
	num_variables = [list_parts[i] for i in range(len(list_parts)) if i not in categorical_indx]


	#using the MinMaxScaler from sklearn.preprocessing, it takes a 2D tab so we reshape list_parts into a 2D tab 
	list_2d = [[feature] for feature in num_variables]
	scaler = MinMaxScaler()
	normalized_num_features = scaler.fit_transform(list_2d) #fit to data, then transform it 

	#we re reshape it again into a list
	normalized_num_features_list = [features[0] for features in normalized_num_features]


	#normalisation of the categorical variables 
	categorical_features= [list_parts[i] for i in categorical_indx]

	#to preserve the orignal order
	normalized_features = []
	num_list_indx= 0

	for i in range( len(list_parts)):
		if i in categorical_indx:
			normalized_features.append(list_parts[i])
		else: 
			normalized_features.append(normalized_num_features_list[num_list_indx])
			num_list_indx+= 1


	return normalized_features



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
						features_list = extract_and_normalize_features(line_parts_float, categorical_features_indeces)
						
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
def target_visualization(x_label, df, kind = 'hist', bins = 30):
	plt.title(f"Distribution of {x_label} frequency")
	#sets a grey background ()
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



"""we chose to use csv files for our data becasuse it is tabular with rows and columns representing different features """

if __name__ == "__main__":
	split_lines('song_data.csv', 56, 'train.csv', 'test.csv', 0.65)
	songs_dict, scores= read_dataset('train.csv')


	"""analyzing the distribution of the song_popularity variable contained in the list scores
	Interpretation: 
		- the curve has a single peak and is very smooth; it indicates a homogenous distribution """
	x_label_popularity = "song_popularity"
	df_popularity = pd.DataFrame({x_label_popularity: scores})

	target_visualization(x_label_popularity, df_popularity, bins= 50)


	#now we want to visualize the distribution by other features 
	df = pd.DataFrame.from_dict(songs_dict, orient= 'index', columns = ['song_duration', 'acousticness', 'danceability', 'energy', 
		'instrumentalness', 'key', 'liveness', 'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature', 'adio_valence'])
	

	x_label_feature = 'acousticness' #example, change this depending on which feature you want to visualize
	target_visualization(x_label_feature, df, kind = 'hist', bins =60) #adjust bins depending on each feature 


	all_features_visualization(df, bins = 10)