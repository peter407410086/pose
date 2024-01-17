import os
import glob
import numpy as np
# 用來計算human 3.6M資料集的檔案共有幾幀

act_names = {
    "02": "Directions",
    "03": "Discussion",
    "04": "Eating",
    "05": "Activities while seated",
    "06": "Greeting",
    "07": "Taking photo",
    "08": "Posing",
    "09": "Making purchases",
    "10": "Smoking",
    "11": "Waiting",
    "12": "Walking",
    "13": "Sitting on chair",
    "14": "Talking on the phone",
    "15": "Walking dog",
    "16": "Walking together"
}

# 指定原始資料夾的路徑
source_folder = "/screamlab/home/peter/Dataset/Human3.6M/train"  # 請將your_source_folder替換為實際的資料夾路徑

# 建立一個字典來追蹤每個 "act" 的總幀數
act_frame_counts = {}
total_frame_count = 0

# 遍歷原始資料夾中的所有.pkl檔案
pkl_files = glob.glob(os.path.join(source_folder, "*.pickle"))

for pkl_file in pkl_files:
    # 提取檔案名稱，例如：s_01_act_02_subact_01_ca_01.pickle
    file_name = os.path.basename(pkl_file)
    
    # 分割檔名，以"_"作為分隔符
    parts = file_name.split("_")
    
    # 提取"act"後面的參數（在這個例子中是"02"）
    act_param = parts[parts.index("act") + 1]

    # 載入.pkl檔案中的資料
    data = np.load(pkl_file, allow_pickle=True)
    
    # 累加幀數到對應的 "act"
    if act_param in act_frame_counts:
        act_frame_counts[act_param] += np.array(data).shape[0]
    else:
        act_frame_counts[act_param] = np.array(data).shape[0]



# 指定原始資料夾的路徑
source_folder = "/screamlab/home/peter/Dataset/mine_3.6/test_angle"  # 請將your_source_folder替換為實際的資料夾路徑

# 遍歷原始資料夾中的所有.pkl檔案
pkl_files = glob.glob(os.path.join(source_folder, "*.pickle"))

for pkl_file in pkl_files:
    # 提取檔案名稱，例如：s_01_act_02_subact_01_ca_01.pickle
    file_name = os.path.basename(pkl_file)
    
    # 分割檔名，以"_"作為分隔符
    parts = file_name.split("_")
    
    # 提取"act"後面的參數（在這個例子中是"02"）
    act_param = parts[parts.index("act") + 1]

    # 載入.pkl檔案中的資料
    data = np.load(pkl_file, allow_pickle=True)
    
    # 累加幀數到對應的 "act"
    if act_param in act_frame_counts:
        act_frame_counts[act_param] += np.array(data).shape[0]
    else:
        act_frame_counts[act_param] = np.array(data).shape[0]

# 將字典按鍵（"act" 參數）進行排序
sorted_act_frame_counts = dict(sorted(act_frame_counts.items()))

# 遍歷所有 "act" 的幀數，並將其相加
for frame_count in sorted_act_frame_counts.values():
    total_frame_count += frame_count

# 列印每個 "act" 的總幀數（按升序），並附帶名稱
for act, frame_count in sorted_act_frame_counts.items():
    act_name = act_names.get(act, "Unknown")  # 如果找不到對應的名稱，使用 "Unknown"
    print(f"act_{act}: {frame_count} frames ({act_name})")

# 列印總幀數
print(f"total frames : {total_frame_count} frames")