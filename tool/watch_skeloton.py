# 印出一筆資料的所有關節座標
import os
import numpy as np

# 定義 joint 字典
joints = {"head": 0, "neck": 1, "rshoulder": 2, "rarm": 3, "rhand": 4,
         "lshoulder": 5, "larm": 6, "lhand": 7, "pelvis": 8,
         "rthigh": 9, "rknee": 10, "rankle": 11, "lthigh": 12, "lknee": 13, "lankle": 14}

jointIndex = {"head":0, "neck":3, "rshoulder":6, "rarm":9, "rhand":12, 
                "lshoulder":15, "larm":18, "lhand":21, "pelvis":24, 
                "rthigh":27, "rknee":30,"rankle":33,"lthigh":36, "lknee":39, "lankle":42}

jointChain = [["neck", "pelvis"], ["head", "neck"],  ["rshoulder", "neck"], ["rarm", "rshoulder"], 
                ["rhand", "rarm"],["rthigh", "pelvis"], ["rknee", "rthigh"], ["rankle", "rknee"],
                ["lshoulder", "neck"], ["larm", "lshoulder"], ["lhand", "larm"], 
                ["lthigh", "pelvis"], ["lknee", "lthigh"], ["lankle", "lknee"]]

jointConnect = [(jointIndex[joints[0]], jointIndex[joints[1]]) for joints in jointChain]

def normalize(data):
        # print("data.shape[0] = ", data.shape)
        data = data.reshape(data.shape[0], int(data.shape[1]/3), 3)
        normal_data = []
        for i, frame in enumerate(data):
            root = (frame[joints['rthigh']]+frame[joints['lthigh']])/2
            data[i, joints['pelvis']] = root
            normal_data.append([])
            for node in frame:
                normal_data[-1].extend(node - root)
        return np.array(normal_data)

def denormalize(normal_data, root):
    data = []
    for i, frame in enumerate(normal_data):
        data.append([])
        for j in range(0, len(frame), 3):
            data[-1].append(frame[j:j+3] + root)
    return np.array(data)

def get_angle(v):
        axis_x = np.array([1,0,0])
        axis_y = np.array([0,1,0])
        axis_z = np.array([0,0,1])

        thetax = axis_x.dot(v)/(np.linalg.norm(axis_x) * np.linalg.norm(v))
        thetay = axis_y.dot(v)/(np.linalg.norm(axis_y) * np.linalg.norm(v))
        thetaz = axis_z.dot(v)/(np.linalg.norm(axis_z) * np.linalg.norm(v))

        return thetax, thetay, thetaz

def calculate_angle(fullbody):
    AngleList = np.zeros_like(fullbody)
    for i, frame in enumerate(fullbody):
        for joint in jointConnect:
            v = frame[joint[0] : joint[0]+3] - frame[joint[1] : joint[1]+3]
            AngleList[i][joint[0] : joint[0]+3] = list(get_angle(v))
    return AngleList

def get_position(v, angles):
        angles = np.arccos(angles)
        r = np.linalg.norm(v)
        x = r*angles[0]
        y = r*angles[1]
        z = r*angles[2]
        return  x,y,z

def calculate_position(fullbody):
    TP_data = np.load('data/split_train_data_with_mirror/pkl/l_dribble_1_0.pkl', allow_pickle=True)
    TP = TP_data[0]
    PosList = np.zeros_like(fullbody)
    for i, frame in enumerate(fullbody):
        for joint in jointConnect:
            v = TP[joint[0] : joint[0]+3] - TP[joint[1] : joint[1]+3]
            angles = frame[joint[0] : joint[0]+3]
            root = PosList[i][joint[1] : joint[1]+3]
            PosList[i][joint[0] : joint[0]+3] = np.array(list(get_position(v, angles))) + root
    return PosList

def calculate_joint_positions(fullbody):
    TP_data = np.load('data/split_train_data_with_mirror/pkl/l_dribble_1_0.pkl', allow_pickle=True)
    TP = TP_data[0]
    PosList = np.zeros_like(fullbody)

    # fullbody裡存的是每一幀的關節cos值
    for i, frame in enumerate(fullbody):
        # 先儲存每一幀的pelvis位置
        PosList[i][joints["pelvis"]*3:(joints["pelvis"]+1)*3] = TP_data[i][joints["pelvis"]*3:(joints["pelvis"]+1)*3]
        # 再計算每一幀的其他關節位置
        for j, joint in enumerate(jointChain):
            print("joint = ", joint)
            child_index = jointIndex[joint[0]] # neck = [3,4,12]
            parent_index = jointIndex[joint[1]] # pelvis = [0,0,0]
            # 計算child和parent的在PosList中的長度
            v = TP[child_index : child_index+3] - TP[parent_index : parent_index+3] # neck - pelvis = [3,4,12] - [0,0,0] = [3,4,12]
            v_length = np.linalg.norm(v) # v_length = 13.0

            PosList[i][child_index : child_index+3] = PosList[i][parent_index : parent_index+3] + get_position(v, frame[j*3: j*3+3])
            break
        break
             
    return PosList

filepath = 'data/split_train_data_with_mirror_2/pkl/l_dribble_1_0.pkl'
if not os.path.exists(filepath):
    print("not found")

with open(filepath, 'rb') as f:
    data = np.load(f, allow_pickle=True)
    print(filepath)
    print("data shape:", data.shape)

    print(f"original neck = {data[0][3:6]}")
    print(f"original pelvis = {data[0][24:27]}")
    data = calculate_angle(data)
    data = calculate_joint_positions(data)
    print(f"position neck = {data[0][3:6]}")

    # root = (data[0][joints["lthigh"]:joints["lthigh"]+3] + data[0][joints["rthigh"]:joints["rthigh"]+3]) / 2
    # data = normalize(data)
    # data = denormalize(data, root)

    # 按照關節的順序印出對應的數值，並對齊格式
    # for joint_name, joint_index in joints.items():
    #     group_indices = data[0][joint_index*3:(joint_index+1)*3]
    #     print(f"{joint_name:9s} : {group_indices}")

    # 遍歷每筆資料並印出 "rhand" 關節的數值
    # for i, motion_data in enumerate(data):
    #     head_indices = motion_data[joints["lhand"] * 3: (joints["lhand"] + 1) * 3]
    #     print(f"frame {i}, lhand: {head_indices}")
    