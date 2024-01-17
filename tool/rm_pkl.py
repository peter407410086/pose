import os
import pickle

pkl_dir = 'data/split_train_data_with_mirror/pkl'
csv_folder = 'data/split_train_data_with_mirror/csv'

for filename in os.listdir(pkl_dir):
    if filename.endswith('.pkl'):
        filepath = os.path.join(pkl_dir, filename)
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
            
            if data.shape[0] > 25 or data.shape[0] < 10:
                os.remove(filepath)
                # 刪除同檔名的 csv 檔案
                csv_filepath = os.path.join(csv_folder, filename.replace('.pkl', '.csv'))
                os.remove(csv_filepath)
