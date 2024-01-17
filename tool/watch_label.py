import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# 設定目標資料夾路徑
folder_path = "data/split_train_data_with_mirror_2/csv"

# 取得目標資料夾中的所有檔案
file_list = os.listdir(folder_path)

# 建立一個空的 DataFrame
df = pd.DataFrame()
cnt = 0
file_cnt = 0

# 讀取每個檔案，並將它們合併成一個 DataFrame
for file_name in tqdm(file_list):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        temp_df = pd.read_csv(file_path)

        # label = 'dribbling_frequency'
        label = 'duration'
        # label = 'bending_angle'
        # label = 'min_height'
        # label = 'first_max_height'
        # label = 'second_max_height'
        if temp_df[label].max() > 24:
            # print(file_name, temp_df[label].max())
            # 刪除 csv 檔案
            if os.path.exists(file_path):
                os.remove(file_path)
            # 刪除同檔名的 pkl 檔案
            pkl_file_path = os.path.join("data/split_train_data_with_mirror_2/pkl", file_name.replace(".csv", ".pkl"))
            if os.path.exists(pkl_file_path):
                os.remove(pkl_file_path)
            cnt += 1
            
            # 刪除的檔案就不用加入 DataFrame
            continue
        file_cnt += 1
        df = pd.concat([df, temp_df], ignore_index=True)
print(f"共刪除 {cnt} 個檔案")
print(f"剩下有 {file_cnt} 個檔案")

# 計算 DataFrame 中每個項目的統計數據
statistics = df.describe()

# 印出每個項目的最高最低值
print(statistics.loc[["min", "max"]])

# 繪製數值分布圖並獲取 Axes 物件
axes = df.hist(bins=50, figsize=(10, 10))

# 為每個子圖添加 x 軸和 y 軸標籤
i = 0
for ax in axes.flatten():
    if i == 0:
        ax.set_xlabel('Right = 1 / Left = 0')
    elif i == 1 or i == 2 or i == 3:
        ax.set_xlabel('Hand Height')
    elif i== 4:
        ax.set_xlabel('times/second')
    elif i== 5:
        ax.set_xlabel('degree')
    elif i== 6:
        ax.set_xlabel('frames')
  
    ax.set_ylabel('amount of motion')
    i+=1

plt.savefig('result/csv.jpg')
