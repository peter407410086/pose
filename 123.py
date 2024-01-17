import os
import numpy as np
import math

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

def translate_pelvis_to_origin(data):
    data = np.array(data).reshape(-1, 15, 3)

    # 找到 "pelvis" 的索引，假設 "pelvis" 在第8個關節
    pelvis_index = joints["pelvis"]

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

def get_position(v, angles):
    r = np.linalg.norm(v)
    x = r*angles[0]
    y = r*angles[1]
    z = r*angles[2]
    
    return  x,y,z

def calculate_angle(fullbody):
    AngleList = np.zeros_like(fullbody)
    for i, frame in enumerate(fullbody):
        for joint in jointConnect:
            v = frame[joint[0]:joint[0]+3] - frame[joint[1]:joint[1]+3]
            AngleList[i][joint[0]:joint[0]+3] = list(get_angle(v))
    return AngleList

def calculate_position(fullbody, TP):
    PosList = np.zeros_like(fullbody)
    for i, frame in enumerate(fullbody):
        for joint in jointConnect:
            v = TP[joint[0]:joint[0]+3] - TP[joint[1]:joint[1]+3]
            angles = frame[joint[0]:joint[0]+3]
            root = PosList[i][joint[1]:joint[1]+3]
            PosList[i][joint[0]:joint[0]+3] = np.array(list(get_position(v, angles)))+root

    return PosList

# def calculate_joint_positions(fullbody):
#     TP_data = np.load('data/split_train_data_with_mirror/pkl/l_dribble_1_0.pkl', allow_pickle=True)
#     TP = TP_data[0]
#     PosList = np.zeros_like(fullbody)

#     # fullbody裡存的是每一幀的關節cos值
#     for i, frame in enumerate(fullbody):
#         # 先儲存每一幀的pelvis位置
#         PosList[i][joints["pelvis"]*3:(joints["pelvis"]+1)*3] = TP_data[i][joints["pelvis"]*3:(joints["pelvis"]+1)*3]
#         # 再計算每一幀的其他關節位置
#         for j, joint in enumerate(jointChain):
#             print("joint = ", joint)
#             child_index = jointIndex[joint[0]] # neck = [3,4,12]
#             parent_index = jointIndex[joint[1]] # pelvis = [0,0,0]
#             # 計算child和parent的在PosList中的長度
#             v = TP[child_index : child_index+3] - TP[parent_index : parent_index+3] # neck - pelvis = [3,4,12] - [0,0,0] = [3,4,12]
#             v_length = np.linalg.norm(v) # v_length = 13.0

#             PosList[i][child_index : child_index+3] = PosList[i][parent_index : parent_index+3] + get_position(v, frame[j*3: j*3+3])
#             break
#         break
             
#     return PosList

# def get_position(v, angles):
#         cos_value = angles[0]
#         angle = math.acos(cos_value)
#         sin_value = math.sin(angle)

#         r = float(np.linalg.norm(v))
#         x = r*cos_value
#         y = r*sin_value
#         return  x,y

# def calculate_joint_positions(fullbody):
#     # TP_data = np.load('data/split_train_data_with_mirror/pkl/l_dribble_1_0.pkl', allow_pickle=True)
#     TP = np.array([3, 4, 0, 0])
#     PosList = np.zeros_like(fullbody)

#     # fullbody裡存的是每一幀的關節cos值
#     for i, frame in enumerate(fullbody):
#         # 先儲存每一幀的pelvis位置
#         # PosList[i][joints["pelvis"]*3:(joints["pelvis"]+1)*3] = TP_data[i][joints["pelvis"]*3:(joints["pelvis"]+1)*3]
#         # 再計算每一幀的其他關節位置
#         for j, joint in enumerate(jointChain):
#             child_index = jointIndex[joint[0]] # neck = [3,4,12]
#             parent_index = jointIndex[joint[1]] # pelvis = [0,0,0]
#             child_index = 0
#             parent_index = 2
#             # 計算child和parent的在PosList中的長度
#             v = TP[child_index : child_index+2] - TP[parent_index : parent_index+2] # neck - pelvis = [3,4,12] - [0,0,0] = [3,4,12]
#             v_length = np.linalg.norm(v) # v_length = 13.0
#             print("v_length = ", v_length)
#             PosList[i][child_index : child_index+2] = PosList[i][parent_index : parent_index+2] + get_position(v_length, frame[j*2: j*2+2])
#             break
#         break
             
#     return PosList

# def get_angle(v):
#         axis_x = np.array([1,0])
#         axis_y = np.array([0,1])

#         thetax = axis_x.dot(v)/(np.linalg.norm(axis_x) * np.linalg.norm(v))
#         thetay = axis_y.dot(v)/(np.linalg.norm(axis_y) * np.linalg.norm(v))

#         return thetax, thetay

# def calculate_angle(fullbody):
#     AngleList = np.zeros_like(fullbody)
#     for i, frame in enumerate(fullbody):
#         v = np.array(frame[0 : 2]) - np.array(frame[2 : 4])
#         AngleList[i][0:2] = list(get_angle(v))
#     return AngleList

data = np.load('data/split_train_data_with_mirror_2/pkl/l_dribble_1_0.pkl', allow_pickle=True)
data = data[0:1]
data = normalize(data)
TP = data[0]

print(f"data = {data[0]}")
data = calculate_angle(data)
print(f"angle = {data[0][0:3]}")
data = calculate_position(data, TP)
print(f"position = {data[0][0:3]}")

# print("--------------------")
# data = np.load('data/split_train_data_with_mirror_2/pkl/r_dribble_01_0.pkl', allow_pickle=True)
# data = data[0:1]
# data = normalize(data)
# TP = data[0]

# print(f"data = {data[0][0:3]}")
# data = translate_pelvis_to_origin(data)
# print(f"translate_pelvis_to_origin = {data[0][0:3]}")
# data = calculate_angle(data)
# print(f"angle = {data[0][0:3]}")
# data = calculate_position(data, TP)
# print(f"position = {data[0][0:3]}")

mpjpe = 0.0
a = np.array([0,0,0,0])
b = np.array([3,4,3,4])
mpjpe = np.mean(np.linalg.norm(a - b))
# for i in range(2):
#     mpjpe += np.mean(np.linalg.norm(a[i] - b[i], axis=0))
print(f"mpjpe = {mpjpe}")