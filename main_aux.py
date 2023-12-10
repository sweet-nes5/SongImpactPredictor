
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler 

def extract_and_normalize_features(list_parts):
	#using the MinMaxScaler from sklearn.preprocessing, it takes a 2D tab so we reshape list_parts into a 2D tab 
	list_2d = [[feature] for feature in list_parts]
	scaler = MinMaxScaler()
	normalized_features = scaler.fit_transform(list_2d) #fit to data, then transform it 
	#we re reshape it again into a list
	normalized_features_list = [features[0] for features in normalized_features]
	return normalized_features_list

""" Parameters: filename (str)
	returns: songs_dict (defaultdict, key= song title, value = features), score(list) """
def read_dataset(filename):
	popularity_scores = []  # song_popularity (what we are seeking to predict), value between 0 and 100 for now ( TODO NORMALIZE THE VALUE INTO A [0,1] VALUE)

	#create a dictionnary {(song title, features)}
	songs_dict = defaultdict()  #À FAIRE 

	try:
		with open(filename, 'r') as input_file: 
			header = input_file.readline() #extract the header
			header_parts = [header.strip() for part in header.strip().split(',')]  #  ['song_name', 'song_popularity', 'song_duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature', 'audio_valence']

			for line in input_file.readlines():
				line_parts = [part.strip() for part in line.strip().split(',')]

				song_name = line_parts[0] #song's title
				#don't add it if it's already in songs_dict
				if song_name not in songs_dict:
					if len(line_parts) == len(header_parts): #check that the data in each line aligns with column name
						features_list = extract_and_normalize_features(line_parts[1:])
						#print(features_list)
						score = features_list[0]
						songs_dict[song_name] = features_list[1:]
						popularity_scores.append(score)					
				else:
					continue

			#print(songs_dict)
			#print(popularity_scores)
			return songs_dict, popularity_scores

	except FileNotFoundError:
		print("File does not exist: {filename}")


if __name__ == "__main__":
	songs_dict, socres= read_dataset('song_data.csv')
