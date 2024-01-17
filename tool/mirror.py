# 把人物的動作左右鏡像
import numpy as np
import os
import pickle
import glob

joints = {
    "head": 0, "neck": 1,
    "rshoulder": 2, "rarm": 3, "rhand": 4,
    "lshoulder": 5, "larm": 6, "lhand": 7,
    "pelvis": 8,
    "rthing": 9, "rknee": 10, "rankle": 11,
    "lthing": 12, "lknee": 13, "lankle": 14
}

# 將左右手以及身體的其他部位進行互換
def mirror_data(data):
    num_frames = data.shape[0]
    mirrored_data = np.copy(data)

    # 鏡像關節坐標在X平面上
    for frame in range(num_frames):
        for joint_name, joint_idx in joints.items():
            mirrored_data[frame, joint_idx, 0] = -data[frame, joint_idx, 0]
            # mirrored_data[frame, joint_idx, 1] = -data[frame, joint_idx, 1]
            # mirrored_data[frame, joint_idx, 2] = -data[frame, joint_idx, 2]

    return mirrored_data


if __name__ == '__main__':
    data_dir = 'data/new_pkl_not_cut'  # 資料夾路徑
    save_dir = 'data/new_pkl_not_cut'  # 存放鏡像數據的資料夾

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
  
    for filename in os.listdir(data_dir):
        if filename[0] == 'm':# 跳過已鏡像後的檔案
            continue

        if filename.endswith('.pkl'):
            data_file = os.path.join(data_dir, filename)
            data = np.load(data_file, allow_pickle=True)
            print(data_file, " => 幀數 = ", np.array(data).shape[0], sep='')
            # print(data.shape)
            data = np.array(data).reshape(-1, 15, 3)
            # 執行鏡像變換
            mirrored_data = mirror_data(data)
            mirrored_data = np.array(mirrored_data).reshape(-1, 45)

            if filename.startswith('r'):
                save_file = os.path.join(save_dir, "mirror_" + filename.replace("r_", "l_"))
            elif filename.startswith('l'):
                save_file = os.path.join(save_dir, "mirror_" + filename.replace("l_", "r_"))
            else:
                save_file = os.path.join(save_dir, "mirror_" + filename)
            # print(save_file)
            # print(mirrored_data.shape)

            # 將鏡像後的數據保存為pkl檔
            with open(save_file, 'wb') as f:
                pickle.dump(mirrored_data, f)
