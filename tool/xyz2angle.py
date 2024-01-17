import os
import pickle
import numpy as np
from tqdm import tqdm

jointIndex = {"head":0, "neck":3, "rshoulder":6, "rarm":9, "rhand":12, 
                "lshoulder":15, "larm":18, "lhand":21, "pelvis":24, 
                "rthigh":27, "rknee":30,"rankle":33,"lthigh":36, "lknee":39, "lankle":42}

joint = {"head":0, "neck":1, "rshoulder":2, "rarm":3, "rhand":4, 
            "lshoulder":5, "larm":6, "lhand":7, "pelvis":8, "rthigh":9, 
            "rknee":10,"rankle":11,"lthigh":12, "lknee":13, "lankle":14}

jointChain = [["neck", "pelvis"], ["head", "neck"],  ["rshoulder", "neck"], ["rarm", "rshoulder"], 
                ["rhand", "rarm"],["rthigh", "pelvis"], ["rknee", "rthigh"], ["rankle", "rknee"],
                ["lshoulder", "neck"], ["larm", "lshoulder"], ["lhand", "larm"], 
                ["lthigh", "pelvis"], ["lknee", "lthigh"], ["lankle", "lknee"]]

jointConnect = [(jointIndex[joint[0]], jointIndex[joint[1]]) for joint in jointChain]

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
                # print("joint = ", joint[0] , joint[0]+3)
                v = frame[joint[0] : joint[0]+3]-frame[joint[1] : joint[1]+3]
                AngleList[i][joint[0] : joint[0]+3] = list(get_angle(v))
        return AngleList

def normalize(data):
        # print("data.shape[0] = ", data.shape, int(data.shape[1]/3))
        data = data.reshape(data.shape[0], int(data.shape[1]/3), 3)
        normal_data = []
        for i, frame in enumerate(data):
            root = (frame[joint['rthigh']]+frame[joint['lthigh']])/2
            data[i, joint['pelvis']] = root
            normal_data.append([])
            for node in frame:
                normal_data[-1].extend(node - root)
        return np.array(normal_data)

def get_position(v, angles):
        r = np.linalg.norm(v)
        x = r*angles[0]
        y = r*angles[1]
        z = r*angles[2]
        return  x,y,z

def calculate_position(fullbody, TP):
        PosList = np.zeros_like(fullbody)
        for i, frame in enumerate(fullbody):
            for joint in jointConnect:
                v = TP[joint[0] : joint[0]+3] - TP[joint[1] : joint[1]+3]
                angles = frame[joint[0] : joint[0]+3]
                root = PosList[i][joint[1] : joint[1]+3]
                PosList[i][joint[0] : joint[0]+3] = np.array(list(get_position(v, angles))) + root

        return PosList

# Load data
input_folder = 'data/split_train_data_with_mirror_2/pkl'
output_folder = 'data/split_train_data_with_mirror_2/angle_pkl'

if not os.path.exists(output_folder):
      os.makedirs(output_folder)

for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith(".pkl"):
        # print(f"input_folder, filename = {input_folder}/{filename}")
        with open(os.path.join(input_folder, filename), 'rb') as f:
            data = pickle.load(f)

        # Normalize
        # data = normalize(data)
        # print("normal_data = ", data[0])

        # position to angle
        angle_data = calculate_angle(data)

        # Angle to position
        # filename = 'data/split_train_data_with_mirror/angle_pkl/l_dribble_1_0.pkl'
        # data = np.load(filename, allow_pickle=True)
        # tp = calculate_position(angle_data, data[0])
        # with open('tp.pkl', 'wb') as f:
        #     pickle.dump(tp, f)

        # Save data
        output_file = os.path.join(output_folder, filename)
        with open(output_file, 'wb') as f:
            pickle.dump(angle_data, f)
