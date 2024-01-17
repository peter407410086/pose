import os
import csv
import numpy as np
import pickle
from tqdm import tqdm

# 用於提取category
def extract_category(filename):
    if "r_dribble" in filename:
        return "right"
    elif "l_dribble" in filename:
        return "left"
    elif "cross" in filename:
        return "cross"
    else:
        return "unknown"

joint = {"head": 0, "neck": 1, "rshoulder": 2, "rarm": 3, "rhand": 4,
         "lshoulder": 5, "larm": 6, "lhand": 7, "pelvis": 8,
         "rthing": 9, "rknee": 10, "rankle": 11, "lthing": 12, "lknee": 13, "lankle": 14}

# 計算運球的最高點高於骨盆多少、運球頻率和運球動作幀數
def calculate_dribbling_parameters(data, hand):
    hand_joint_index = 4 if hand == "right" else 7  # 右手為4，左手為7
    y_coordinates = data[:, hand_joint_index * 3 + 1]  # 提取手部Y軸座標
    # 初始化參數
    dribbling_frames = []  # 用來存儲運球的起始幀數、結束幀數和持續幀數
    is_dribbling = False  # 用來追蹤運球狀態
    first_max_height = y_coordinates[0]
    sec_max_height = y_coordinates[0]  # 運球的最大高度
    min_height = y_coordinates[0]  # 運球的最小高度
    pre_y = y_coordinates[0]  # 上一幀的y坐標
    dribbling_start_frame = 0  # 運球開始的幀數
    waiting_for_max = True  # 用來追蹤是否等待最高點

    for i, y in enumerate(y_coordinates):
        if is_dribbling:
            if y <= min_height:  # 手正在下降
                # print("手正在下降")
                min_height = y
                sec_max_height = y
            elif y > sec_max_height:  # 手正在回到最高點
                # print("手正在回到最高點")
                sec_max_height = y
            elif y < sec_max_height:  # 手開始往下的瞬間
                # print("手開始往下的瞬間")
                dribbling_frames.append((dribbling_start_frame, i - 1, i - dribbling_start_frame))
                # print("dribbling_frames = ", dribbling_frames)
                is_dribbling = False
                break
        elif y > first_max_height and waiting_for_max:  # 手正在上升，且等待最高點
            # print("等待最高點")
            first_max_height = y
        elif y < first_max_height and waiting_for_max:  # 手開始往下的瞬間
            # print("手開始往下的瞬間")
            is_dribbling = True
            first_max_height = pre_y
            min_height = y
            dribbling_start_frame = i-1
            waiting_for_max = False
        
        is_dribbling = True
        pre_y = y  #更新上一幀的y坐標
    # print(f"first_max = {first_max_height}, min = {min_height}, second_max = {sec_max_height}")

    # 檢查最後一次運球是否結束，並執行切割操作
    # if is_dribbling:   
    #     dribbling_frames.append((dribbling_start_frame, len(y_coordinates) - 1, len(y_coordinates) - dribbling_start_frame))

    # 計算運球段的持續幀數
    for i in range(len(dribbling_frames)):
        start_frame, end_frame, _ = dribbling_frames[i]
        duration = end_frame - start_frame # duration沒有包含最後一幀，所以不用加1
        dribbling_frames[i] = (start_frame, end_frame, duration)

    # 轉換為每秒的運球次數（假設每幀間隔為1/30秒）
    if dribbling_frames:
        # print(f"dribbling_frames[-1][1] = {dribbling_frames[-1][1]}")
        # print(f"dribbling_frames[0][0] = {dribbling_frames[0][0]}")
        dribbling_frequency = len(dribbling_frames) / (dribbling_frames[-1][1] - dribbling_frames[0][0]) * 30
    else:
        dribbling_frequency = 0

    return first_max_height, sec_max_height, min_height, dribbling_frequency, dribbling_frames

# 計算彎腰角度的函數
def calculate_bending_angle(pelvis_coords, neck_coords):
    # print(f"pelvis_coords = {pelvis_coords}, neck_coords = {neck_coords}")
    # 計算 pelvis 和 neck 之間的向量
    vector_pelvis_to_neck = neck_coords - pelvis_coords

    # 建立一個參考向量，例如 [0, 1, 0]，代表垂直於地面的向量，根據你的座標系統可能需要調整
    reference_vector = np.array([0, 1, 0])

    # 計算兩個向量之間的內積
    dot_product = np.dot(vector_pelvis_to_neck, reference_vector)

    # 計算兩個向量的長度
    magnitude_pelvis_to_neck = np.linalg.norm(vector_pelvis_to_neck)
    magnitude_reference = np.linalg.norm(reference_vector)

    # 計算夾角的弧度
    cosine_similarity = dot_product / (magnitude_pelvis_to_neck * magnitude_reference)
    angle_radians = np.arccos(np.clip(cosine_similarity, -1, 1))

    # 將弧度轉換為角度
    angle_degrees = np.degrees(angle_radians)
    # print(f"angle_degrees = {angle_degrees}")
    return angle_degrees

# 用於儲存骨架資訊為pkl檔
def save_skeleton_info_to_pkl(data, prefix, index, output_folder):
    # 構建檔案名稱
    pkl_data_folder = os.path.join(output_folder, "pkl")
    os.makedirs(pkl_data_folder, exist_ok=True)

    output_filename = f"{prefix}_{index}.pkl"
    output_file_path = os.path.join(pkl_data_folder, output_filename)
    
    # 儲存骨架資訊到pkl檔
    with open(output_file_path, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

# 函數用於儲存相應的txt檔
def save_info_to_txt(hand, first_max_height, second_max_height, min_height, dribbling_frequency, frames_range, bending_angle, prefix, index, output_folder):
    # 構建檔案名稱
    txt_data_folder = os.path.join(output_folder, "txt")
    os.makedirs(txt_data_folder, exist_ok=True)
    txt_output_file_path = os.path.join(txt_data_folder, f"{prefix}_{index}.txt")

     # 轉換手的字串為數字
    hand = 1 if hand == 'right' else 0
    
    # 儲存txt檔
    with open(txt_output_file_path, "w") as txt_file:
        txt_file.write(f"Dribbling Hand: {hand}\n")
        txt_file.write(f"First Max Height: {first_max_height}\n")
        txt_file.write(f"Min Height: {min_height}\n")
        txt_file.write(f"Second Max Height: {second_max_height}\n")
        txt_file.write(f"Dribbling Frequency: {dribbling_frequency} times/sec\n")
        txt_file.write(f"Bending Angle: {bending_angle} degrees\n")
        # txt_file.write(f"Frames Range: {frames_range[0]} to {frames_range[1]}\n")
        txt_file.write(f"Duration: {frames_range[2]} frames\n")

# 函數用於儲存相應的csv檔
def save_info_to_csv(hand, first_max_height, second_max_height, min_height, dribbling_frequency, frames_range, bending_angle, prefix, index, output_folder):
    # 構建檔案名稱
    csv_data_folder = os.path.join(output_folder, "csv")
    os.makedirs(csv_data_folder, exist_ok=True)
    csv_output_file_path = os.path.join(csv_data_folder, f"{prefix}_{index}.csv")
    
    # 轉換手的字串為數字
    hand = 1 if hand == 'right' else 0
    # 儲存csv檔
    with open(csv_output_file_path, mode='w') as csv_file:
        fieldnames = ['hand', 'first_max_height', 'min_height', 'second_max_height', 'dribbling_frequency', 'bending_angle', 'duration']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'hand': hand, 'first_max_height': first_max_height, 'min_height': min_height, 'second_max_height': second_max_height, 
                         'dribbling_frequency': dribbling_frequency, 'bending_angle': bending_angle, 'duration': frames_range[2]})
        
# 指定你的資料夾路徑
data_folder = "data/new_pkl_not_cut"

# 創建目錄來存放切割後的檔案
output_folder = "data/split_train_data_with_mirror_2"
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

# 遍歷資料夾中的所有檔案
cnt = 0
for filename in tqdm(os.listdir(data_folder)):
    # if cnt == 1:
    #     break
    # 先不管換手運球
    if filename.startswith('cross') or filename.startswith('mirror_cross'):
        continue

    if filename.endswith(".pkl"):
        hand = extract_category(filename)
        hand_joint_index = 4 if hand == "right" else 7  # 右手為4，左手為7

        # 讀取pkl檔案
        data = np.load(os.path.join(data_folder, filename), allow_pickle=True)
        num_frames, num_joints = data.shape

        # 初始化運球段的起始幀為0
        dribbling_start_frame = 0
        while dribbling_start_frame < num_frames:
            # print到哪個檔案的哪個幀數
            # print(f"filename = {filename}, frame = {dribbling_start_frame}")
            # 計算運球的最高點高於骨盆多少、運球頻率和運球動作幀數
            first_max_height, second__max_height, min_height, dribbling_frequency, dribbling_frames = calculate_dribbling_parameters(data[dribbling_start_frame:], hand)
            # print(f"Max Height: {max_height}")
            # print(f"Min Height: {min_height}")
            # print(f"Dribbling Frequency: {dribbling_frequency} times/sec\n")
            # print(f"drbbling_start_frame = {dribbling_start_frame}\n")
            # print(f"Dribbling Frames: {dribbling_frames}\n")

            # 如果沒有運球段，則跳出循環
            if not dribbling_frames:
                # print("沒有運球段")
                break

            # 取得運球段的幀數範圍
            start_frame, end_frame, duration = dribbling_frames[0]

            # if duration < 10 or duration > 26:
            #     # 更新運球段的起始幀
            #     dribbling_start_frame += end_frame
            #     # print("dribbling_start_frame = ", dribbling_start_frame)
            #     continue

            # 計算彎腰角度
            pelvis_coords = data[dribbling_start_frame + start_frame, joint["pelvis"] * 3:joint["pelvis"] * 3 + 3]
            neck_coords = data[dribbling_start_frame + start_frame, joint["neck"] * 3:joint["neck"] * 3 + 3]
            bending_angle = calculate_bending_angle(pelvis_coords, neck_coords)

            # 創建子目錄以存放不同類別的檔案
            category_output_folder = os.path.join(output_folder)
            os.makedirs(category_output_folder, exist_ok=True)

            # 儲存骨架資訊為 pkl 檔，保留原始前綴
            save_skeleton_info_to_pkl(data[dribbling_start_frame:dribbling_start_frame + end_frame], filename.split('.')[0], dribbling_start_frame, category_output_folder)
            
            # 儲存相應的 csv 檔
            save_info_to_csv(hand, first_max_height, second__max_height, min_height, dribbling_frequency, (dribbling_start_frame + start_frame, dribbling_start_frame + end_frame, duration), bending_angle, filename.split('.')[0], dribbling_start_frame, category_output_folder)

            # 儲存相應的 txt 檔
            # save_info_to_txt(hand, first_max_height, second__max_height, min_height, dribbling_frequency, (dribbling_start_frame + start_frame, dribbling_start_frame + end_frame, duration), bending_angle, filename.split('.')[0], dribbling_start_frame, category_output_folder)
            
            # 更新運球段的起始幀
            dribbling_start_frame += end_frame
            # print("dribbling_start_frame = ", dribbling_start_frame)    

    cnt+=1
print(f"count = {cnt}")
print("骨架資訊已切割並存入目錄")
