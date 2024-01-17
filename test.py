import pickle
import numpy as np
import os, re, random
import torch
from utils import jointIndex, jointConnect
# get_data 將所有的資料包裝成很長的序列，所有檔案的資料都在同一個 data['x'] 內，因此檔案數量不會和 mpjpe 個數相同
def get_data(dataset, dir, ca, inp_len, out_len, randomInput): # for train 
    filepath = os.path.join("../Dataset", dataset, dir)
    files = os.listdir(filepath)
    files.sort()
    
    traindatas = []
    ca = ca.split("_")
    total_frame = 0
    
    choose_files = []
    for file in files:
        if re.split("_|\.", file)[-2] in ca :
            choose_files.append(file)
            with open(os.path.join(filepath, file), 'rb') as f:
                data = pickle.load(f)
            total_frame += len(data)
            for i in range(0, len(data)-(inp_len+out_len), 5):
                if randomInput:
                    rand = random.randint(0, int(inp_len/2))
                else:
                    rand = int(inp_len/2)

                input_data = np.array(data[i:i+(rand*2)+out_len])
                output_data =  np.array(data[i:i+(rand*2)+out_len])
                input_data[rand : rand+out_len] = 1
                traindatas.append((input_data, output_data))
    print("total frames: ",total_frame)
    with open("datas_name.txt",'w') as f:
        f.write('\n'.join(choose_files))    
    data = {"x":[], "y":[]}
    for i, (x,y) in enumerate(traindatas):
        data["x"].append(torch.tensor(x.astype("float32")))        
        data["y"].append(torch.tensor(y.astype("float32")))

    data["x"] = torch.nn.utils.rnn.pad_sequence(data["x"], batch_first=True)
    data["y"] = torch.nn.utils.rnn.pad_sequence(data["y"], batch_first=True)
    
    return data

def get_part_data(dataset, dir, ca, part, inp_len, out_len, randomInput, joint_def): # for train 
    filepath = os.path.join("../Dataset", dataset, dir)
    files = os.listdir(filepath)
    files.sort()
    traindatas = []
    ca = ca.split("_")
    for file in files:
        if re.split("_|\.", file)[-2] in ca :
            with open(os.path.join(filepath, file), 'rb') as f:
                data = pickle.load(f)
            data = joint_def.cat_numpy(part, data)
            for i in range(0, len(data)-(inp_len+out_len), 5):
                if randomInput:
                    rand = random.randint(0, int(inp_len/2))
                else:
                    rand = int(inp_len/2)

                input_data = np.array(data[i:i+(rand*2)+out_len])
                output_data =  np.array(data[i: i+(rand*2)+out_len])
                input_data[rand: rand+out_len] = 1
                traindatas.append((input_data, output_data))

    data = {"x":[], "y":[]}
    for i, (x,y) in enumerate(traindatas):
        data["x"].append(torch.tensor(x.astype("float32")))        
        data["y"].append(torch.tensor(y.astype("float32")))

    data["x"] = torch.nn.utils.rnn.pad_sequence(data["x"], batch_first=True)
    data["y"] = torch.nn.utils.rnn.pad_sequence(data["y"], batch_first=True)

    return data

def get_single_data(dir, filename, file):
    if dir != "":
        filepath = os.path.join("../Dataset", dir, filename, file)
    else:
        filepath = os.path.join("../Dataset", file)

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def normalize(fullbody):
    normal = np.zeros_like(fullbody)
    hips = jointIndex['hips']
    for i, frame in enumerate(fullbody):
        for joint in jointIndex.values():
            normal[i][joint:joint+3] = frame[joint:joint+3] - frame[hips:hips+3]
    return normal

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

def classify(fullbody):
    body = []
    larm = []
    lleg = []
    rarm = []
    rleg = []
    for data_frame in fullbody:
        body.append([data_frame[0],data_frame[1],data_frame[2],data_frame[3],data_frame[4],data_frame[5],data_frame[24],data_frame[25],data_frame[26]])
        rarm.append([data_frame[24],data_frame[25],data_frame[26],data_frame[3],data_frame[4],data_frame[5],data_frame[6],data_frame[7],data_frame[8],data_frame[9],data_frame[10],data_frame[11],data_frame[12],data_frame[13],data_frame[14]])
        rleg.append([data_frame[3],data_frame[4],data_frame[5],data_frame[24],data_frame[25],data_frame[26],data_frame[27],data_frame[28],data_frame[29],data_frame[30],data_frame[31],data_frame[32],data_frame[33],data_frame[34],data_frame[35]])
        larm.append([data_frame[24],data_frame[25],data_frame[26],data_frame[3],data_frame[4],data_frame[5],data_frame[15],data_frame[16],data_frame[17],data_frame[18],data_frame[19],data_frame[20],data_frame[21],data_frame[22],data_frame[23]])
        lleg.append([data_frame[3],data_frame[4],data_frame[5],data_frame[24],data_frame[25],data_frame[26],data_frame[36],data_frame[37],data_frame[38],data_frame[39],data_frame[40],data_frame[41],data_frame[42],data_frame[43],data_frame[44]])
    
    return np.array(body), np.array(larm), np.array(lleg), np.array(rarm), np.array(rleg)

if __name__ == '__main__':
    dir = "Mixamo"
    filename = "test_angle"
    file = "down_w_act_01_ca_01"
    train_ca = "01_02_03_04_1_2_3_4"
    inp_len = 20
    out_len = 30
    part = 'leftleg'

    train_data = get_data(dir, filename, ca=train_ca,
                                    inp_len=inp_len, out_len=out_len, randomInput=False)
