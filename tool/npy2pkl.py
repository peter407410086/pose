import os
import numpy as np
import pickle

def convert_npy_to_pkl(input_folder, output_folder):
    # 檢查輸出資料夾是否存在，如果不存在，則創建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 獲取輸入資料夾中的所有npy檔案
    npy_files = [f for f in os.listdir(input_folder) if f.endswith(".npy")]

    # 依序處理每個npy檔案
    for npy_file in npy_files:
        npy_path = os.path.join(input_folder, npy_file)
        pkl_file = os.path.splitext(npy_file)[0] + ".pkl"  # 使用相同的檔名，但副檔名為pkl
        pkl_path = os.path.join(output_folder, pkl_file)

        # 讀取npy檔案
        data = np.load(npy_path)
        print("data = ", data.shape)

        # 儲存為pkl檔案
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump(data, pkl_file)

        print(f"已轉換並儲存檔案：{pkl_path}")

if __name__ == "__main__":
    input_dir = 'lhand_dribble_2'
    input_folder = "data/npy/lhand_dribble/" + input_dir  # 設定為你的輸入資料夾路徑
    output_folder = "/home/peter/3d_motion_generator/data/pkl"  # 設定為你想要儲存pkl檔案的資料夾路徑
    convert_npy_to_pkl(input_folder, output_folder)
