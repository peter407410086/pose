import os
import pickle

# 資料夾路徑
folder_path = 'data/split_train_data_with_mirror/pkl'

# 初始化計數器
lhand_frames = 0
rhand_frames = 0
cross_frames = 0
mirror_lhand_frames = 0
mirror_rhand_frames = 0
mirror_cross_frames = 0

# 列舉資料夾中的所有檔案
for filename in os.listdir(folder_path):
    if filename.endswith(".pkl"):
        # 讀取pkl檔案
        with open(os.path.join(folder_path, filename), 'rb') as file:
            data = pickle.load(file)
        if data.shape[0] >= 24:
            print(filename)

        # 檢查檔名開頭並統計幀數
        if filename.startswith('l'):
            lhand_frames += len(data)
        elif filename.startswith('mirror_l'):
            mirror_lhand_frames += len(data)
        elif filename.startswith('r'):
            rhand_frames += len(data)
        elif filename.startswith('mirror_r'):
            mirror_rhand_frames += len(data)
        elif filename.startswith('cross'):
            cross_frames += len(data)
        elif filename.startswith('mirror_cross'):
            mirror_cross_frames += len(data)

# 輸出統計結果
print(f"'lhand_dribble': {lhand_frames} frames")
print(f"'rhand_dribble': {rhand_frames} frames")
print(f"'crossover': {cross_frames} frames")
print(f"'lhand_dribble'+'mirror_lhand_dribble': {lhand_frames + mirror_lhand_frames} frames")
print(f"'rhand_dribble'+'mirror_rhand_dribble': {rhand_frames + mirror_rhand_frames} frames")
print(f"'cross'+'mirror_cross': {cross_frames + mirror_cross_frames} frames")
