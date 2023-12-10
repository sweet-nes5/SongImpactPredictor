
from collections import defaultdict



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
						score = int(line_parts[1]) # storing the song's popularity rate in score (TODO NORMALIZE)
						features = [float(i) for i in line_parts[1:]]
						songs_dict[song_name] = features
						popularity_scores.append(score)
				else:
					continue


			print(songs_dict)

			return songs_dict, popularity_scores

	except FileNotFoundError:
		print("File does not exist: {filename}")


if __name__ == "__main__":
	songs_dict, socres= read_dataset('song_data.csv')
