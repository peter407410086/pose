import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Specify the directory where the label files are located
label_dir = 'data/split_train_data_with_mirror/csv'

# Initialize a dictionary to store the scalers for each column
scalers = {
    'first_max_height': MinMaxScaler(),
    'min_height': MinMaxScaler(),
    'second_max_height': MinMaxScaler(),
    'dribbling_frequency': MinMaxScaler(),
    'bending_angle': MinMaxScaler(),
}

# Initialize a dictionary to store all data
all_data = {key: [] for key in scalers.keys()}
all_data['duration'] = []
all_data['hand'] = []

# Loop through all files in the directory to collect all data
for filename in os.listdir(label_dir):
    if filename.endswith('.csv'):  # Assuming the label files are in CSV format
        # Construct the full file path
        file_path = os.path.join(label_dir, filename)

        # Load the label file
        df = pd.read_csv(file_path)
        
        # Append the data to the corresponding list
        for column in df.columns:
            if column in scalers.keys():
                all_data[column].append(df[column].values)
            elif column == 'duration':
                all_data['duration'].append(df[column].values)
            elif column == 'hand':
                all_data['hand'].append(df[column].values)

# Fit the scalers to the corresponding data
for column in scalers.keys():
    scalers[column].fit(np.concatenate(all_data[column]).reshape(-1, 1))

# Loop through all files in the directory again to normalize and save
output_dir = 'data/split_train_data_with_mirror/normalized_csv'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(label_dir):
    if filename.endswith('.csv'):  # Assuming the label files are in CSV format
        # Construct the full file path
        input_file_path = os.path.join(label_dir, filename)
        output_file_path = os.path.join(output_dir, filename)

        # Load the label file
        df = pd.read_csv(input_file_path)

        # Normalize the data
        for column in df.columns:
            if column in scalers.keys():
                df[column] = scalers[column].transform(df[column].to_numpy().reshape(-1, 1))
            elif column == 'duration' or column == 'hand':
                all_data[column].append(df[column].values)

        # Save the normalized data back to the file
        df.to_csv(output_file_path, index=False)
