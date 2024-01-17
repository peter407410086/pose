import csv
import os
import pickle
from tqdm import tqdm

joints = {"head": 0, "neck": 1, "rshoulder": 2, "rarm": 3, "rhand": 4,
         "lshoulder": 5, "larm": 6, "lhand": 7, "pelvis": 8,
         "rthing": 9, "rknee": 10, "rankle": 11, "lthing": 12, "lknee": 13, "lankle": 14}

def calculate_min_height_frame(hand_positions):
    min_height = float('inf')
    min_height_frame = None
    for i, position in enumerate(hand_positions):
        if position[1] < min_height:
            min_height = position[1]
            min_height_frame = i
    return min_height_frame

def save_min_height_frame_to_csv(min_height_frame, second_max_height, filename):
    filename = filename.replace('.pkl', '.csv')
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['first_max_height_frame', 'min_height_frame', 'second_max_height_frame'])
        writer.writerow([0, min_height_frame, second_max_height])

# file_folder = 'test/pkl'
# save_folder = 'test/min_height_frame'
file_folder = 'test/pkl'
save_folder = 'test/key_frame'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 讀取資料夾所有pkl檔
for file in tqdm(os.listdir(file_folder)):
    # 讀取pkl檔
    file_dir = os.path.join(file_folder, file)
    with open(file_dir, 'rb') as f:
        data = pickle.load(f)

    hand = ''
    if file.startswith('l') or file.startswith('mirror_l'):
        hand = 'lhand'
    elif file.startswith('r') or file.startswith('mirror_r'):
        hand = 'rhand'

    # 取出左手的所有位置，進來的資料格式是(frames, 45)
    hand_positions = data[:, joints[hand]*3:joints[hand]*3+3]
    
    second_max_height = data.shape[0] - 1
    # 計算最低點的frame
    min_height_frame = calculate_min_height_frame(hand_positions)
    # print(f"min_height_frame = {min_height_frame}")
    # print(f"second_max_height = {second_max_height}")
    # 儲存的檔名跟原本的檔名一樣，並存到test/min_height_frame
    save_dir = os.path.join(save_folder, file)
    save_min_height_frame_to_csv(min_height_frame, second_max_height, save_dir)