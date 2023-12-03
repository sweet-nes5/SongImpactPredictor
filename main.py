import os
directory = './'

def read_dataset(filename):
    features = []  # relevent features such as : audio_valence, acousticness, danceability, energy, loudness
    labels = [] # song_popularity (what we are seeking to predict)
    filename = os.path.join(directory,filename)
    with open (filename, 'r') as file:
        header = file.readline()  # extract the header
        header_parts = [part.strip() for part in header.strip().split(',')]
        print("Header:", header_parts)
        for line in file.readlines():
            line_parts = [part.strip() for part in line.strip().split(',')]
            #print(line_parts)
            if len(line_parts) == len(header_parts):# check that the data in each line aligns with column name
                label = int(line_parts[1])  #storing the song popularity rate in label
                feature = [float(line_parts[i]) for i in [3, 4, 5, 9, 14]] # extracting the relevent features
                features.append(feature)
                labels.append(label)
    #print("Features:", features)
    #print("Labels:", labels)
    return features, labels

features, labels = read_dataset('song_data.csv')



