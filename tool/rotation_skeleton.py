import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
from tqdm import tqdm

joint = {"head":0, "neck":1, "rshoulder":2, "rarm":3, "rhand":4, 
            "lshoulder":5, "larm":6, "lhand":7, "pelvis":8, "rthigh":9, 
            "rknee":10,"rankle":11,"lthigh":12, "lknee":13, "lankle":14}
# 每個關節點都移動 把骨盆移到原點
def translate_pelvis_to_origin(data):
    data = np.array(data).reshape(-1, 15, 3)

    # 找到 "pelvis" 的索引，假設 "pelvis" 在第8個關節
    pelvis_index = joint["pelvis"]

    # 獲取 pelvis 的座標
    pelvis_coordinates = data[:, pelvis_index]

    # 計算平移向量
    translation_vector = -pelvis_coordinates[:, np.newaxis, :]

    # 對每個關節的座標進行平移
    data += translation_vector

    # 最後，pelvis 的座標應該變為 [0, 0, 0]
    data[:, pelvis_index] = [0, 0, 0]

    # 打印平移後的第一幀座標
    # for j, joint_coordinates in enumerate(data[0]):
    #     print(f"Joint {j}: {joint_coordinates}")

    return np.array(data).reshape(-1, 45)

def normalize(data):
        # print("data.shape[0] = ", data.shape)
        data = data.reshape(data.shape[0], int(data.shape[1]/3), 3)
        normal_data = []
        for i, frame in enumerate(data):
            root = (frame[joint['rthigh']]+frame[joint['lthigh']])/2
            data[i, joint['pelvis']] = root
            normal_data.append([])
            for node in frame:
                normal_data[-1].extend(node - root)
        return np.array(normal_data)

# 旋轉骨架用的
# for i in range(1):
#     print("i = ",i)
#     file_name = 'move_output_1'
#     # 載入npy檔案
#     # data = np.load('img/'+file_name+'.npy')
#     data = np.load('output_1.npy')

#     data = np.array(data).reshape(-1, 15, 3)
#     print(data[0])
#     # for i in data:
#     #     for j in i:
#     #         j[1] = -j[1]
    
#     # print("data = ", data[0])

#     # 使用平移函數
#     data = translate_pelvis_to_origin(data)
#     data = np.array(data).reshape(-1, 45)
#     # print(data[0])

#     # 設定轉換參數
#     elev = 180
#     azim = -180

#     # 計算投影矩陣
#     proj_matrix = np.array([
#         [1, 0, 0],
#         [0, np.cos(np.radians(elev)), -np.sin(np.radians(elev))],
#         [0, np.sin(np.radians(elev)), np.cos(np.radians(elev))]
#     ])
#     proj_matrix = np.matmul(
#         np.array([
#             [np.cos(np.radians(azim)), -np.sin(np.radians(azim)), 0],
#             [np.sin(np.radians(azim)), np.cos(np.radians(azim)), 0],
#             [0, 0, 1]
#         ]),
#         proj_matrix
#     )

#     # 將 3D 向量轉換成 2D 平面上的點
#     # proj_data = np.matmul(data.reshape(-1, 3), proj_matrix.T)
#     # proj_data = proj_data.reshape(data.shape[0], -1)
#     # print("proj_data = ", proj_data.shape)

#     # normal_data = normalize(data)
#     # print("normalize_data = \n",data[0])

#     # 將資料儲存到新的npy檔案中
#     np.save(file_name +'.npy', data.reshape((-1, 45)))
#     break

cnt = 0
data_dir = 'data/split_train_data_with_mirror/pkl'
save_dir = 'data/split_train_data_with_mirror/normalized_pkl'
for filename in tqdm(os.listdir(data_dir)):  
    # if(filename[0] == 'm'  or filename[0] == 'l' ):
    #      continue      
    data = np.load(os.path.join(data_dir, filename), allow_pickle=True)
    # print(filename, " => 幀數 = ", np.array(data).shape[0], sep='')
    # data = translate_pelvis_to_origin(data)
    # print("data = ", data[0])
    data = normalize(data)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, filename)
    # save_file = 'img/r_end.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(data, f)

    cnt += 1

print("cnt = ", cnt)