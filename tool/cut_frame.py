import os
import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import glob

if __name__ == '__main__':
    pkl_files = glob.glob('data/pkl_not_cut/*.pkl')

    save_dir = 'data/new_pkl_not_cut'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for next_file in tqdm(pkl_files):
        if not os.path.exists(next_file):
            print("無檔案 => ", next_file)
            continue

        data = np.load(next_file, allow_pickle=True)
        # print(next_file, " => 幀數 = ", np.array(data).shape[0], sep='')

        # 把next_file的檔名分割成路徑、檔名、副檔名
        path, filename = os.path.split(next_file)
        # print("filename = ", filename)
        save_path = os.path.join(save_dir, filename)
        print("save_path = ", save_path)

        with open(save_path, 'wb') as f:
            pickle.dump(data[5:, :], f)
        
