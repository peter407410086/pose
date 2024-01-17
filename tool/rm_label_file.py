import os

pkl_folder = 'data/split_train_data_with_mirror/pkl'
csv_folder = 'data/split_train_data_with_mirror/csv'

# 刪除 pkl 檔案
# for pkl_file in os.listdir(pkl_folder):
#     pkl_file_path = os.path.join(pkl_folder, pkl_file)
#     csv_file_path = os.path.join(csv_folder, pkl_file.replace('.pkl', '.csv'))
    
#     if not os.path.exists(csv_file_path):
#         print(csv_file_path, "不存在")
#         os.remove(pkl_file_path)
        
# 刪除 csv 檔案
for csv_file in os.listdir(csv_folder):
    csv_file_path = os.path.join(csv_folder, csv_file)
    pkl_file_path = os.path.join(pkl_folder, csv_file.replace('.csv', '.pkl'))
    if not os.path.exists(pkl_file_path):
        print(pkl_file_path, "不存在")
        # os.remove(csv_file_path)