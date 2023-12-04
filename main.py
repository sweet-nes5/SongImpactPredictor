import os
directory = './'


def read_dataset(filename):
    features = []  # relevant features such as : audio_valence, acousticness, danceability, energy, loudness
    labels = []  # song_popularity (what we are seeking to predict)
    filename = os.path.join(directory, filename)
    unique_name_song = set()  # to eliminate the duplicate songs
    song_line = 0  # to count the number of songs in total
    with open(filename, 'r') as file:
        header = file.readline()  # extract the header
        header_parts = [part.strip() for part in header.strip().split(',')]
        print("Header:", header_parts)
        for line in file.readlines():
            line_parts = [part.strip() for part in line.strip().split(',')]
            song_name = line_parts[0]
            song_line += 1
            if song_name in unique_name_song:
                continue
            else:
                unique_name_song.add(song_name)
            # print(line_parts)

            if len(line_parts) == len(header_parts):  # check that the data in each line aligns with column name
                label = int(line_parts[1])   # storing the song popularity rate in label
                feature = [float(line_parts[i]) for i in [3, 4, 5, 9, 14]]  # extracting the relevant features
                features.append(feature)
                labels.append(label)
    # print("number of name songs before :", song_line)
    # print("number of songs after :", len(unique_name_song))
    # print("Features:", features)
    # print("Labels:", labels)
    return features, labels


features_extraction, labels_extraction = read_dataset('song_data.csv')
