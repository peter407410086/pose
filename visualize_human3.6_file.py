import os
import pickle
import argparse
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.mixture import GaussianMixture
import torch
from tqdm import tqdm

joint = {"head": 0, "neck": 1, "rshoulder": 2, "rarm": 3, "rhand": 4,
         "lshoulder": 5, "larm": 6, "lhand": 7, "pelvis": 8,
         "rthing": 9, "rknee": 10, "rankle": 11, "lthing": 12, "lknee": 13, "lankle": 14}

spine = ['pelvis', 'neck', 'head']
r_arm = ['neck', 'rshoulder', 'rarm', 'rhand']
l_arm = ['neck', 'lshoulder', 'larm', 'lhand']
r_leg = ['pelvis', 'rthing', 'rknee', 'rankle']
l_leg = ['pelvis', 'lthing', 'lknee', 'lankle']
body = [spine, r_leg, l_leg, r_arm, l_arm]

jointIndex = {"head":0, "neck":3, "rshoulder":6, "rarm":9, "rhand":12, 
                "lshoulder":15, "larm":18, "lhand":21, "pelvis":24, 
                "rthigh":27, "rknee":30,"rankle":33,"lthigh":36, "lknee":39, "lankle":42}

jointChain = [["neck", "pelvis"], ["head", "neck"],  ["rshoulder", "neck"], ["rarm", "rshoulder"], 
                ["rhand", "rarm"],["rthigh", "pelvis"], ["rknee", "rthigh"], ["rankle", "rknee"],
                ["lshoulder", "neck"], ["larm", "lshoulder"], ["lhand", "larm"], 
                ["lthigh", "pelvis"], ["lknee", "lthigh"], ["lankle", "lknee"]]

jointConnect = [(jointIndex[joints[0]], jointIndex[joints[1]]) for joints in jointChain]

def normalized(v):
    return v / np.linalg.norm(v)

def calculate_angle(fullbody):
    def get_angle(v):
        axis_x = np.array([1,0,0])
        axis_y = np.array([0,1,0])
        axis_z = np.array([0,0,1])

        thetax = axis_x.dot(v)/(np.linalg.norm(axis_x) * np.linalg.norm(v))
        thetay = axis_y.dot(v)/(np.linalg.norm(axis_y) * np.linalg.norm(v))
        thetaz = axis_z.dot(v)/(np.linalg.norm(axis_z) * np.linalg.norm(v))

        return thetax, thetay, thetaz
    print(fullbody.shape)
    fullbody = fullbody.reshape(-1, 45)
    AngleList = np.zeros_like(fullbody)
    for i, frame in enumerate(fullbody):
        for joint in jointConnect:
            # print("joint = ", joint[0] , joint[0]+3)
            v = frame[joint[0] : joint[0]+3]-frame[joint[1] : joint[1]+3]
            AngleList[i][joint[0] : joint[0]+3] = list(get_angle(v))
    return AngleList

def calculate_position(fullbody):
    def get_position(v, angles):
        r = np.linalg.norm(v)
        x = r*angles[0]
        y = r*angles[1]
        z = r*angles[2]
        return  x,y,z

    TP_data = np.load('data/TPose/s_01_act_02_subact_01_ca_01.pickle', allow_pickle=True)
    TP = TP_data[0]
    PosList = np.zeros_like(fullbody)
    for i, frame in enumerate(fullbody):
        for joint in jointConnect:
            v = TP[joint[0] : joint[0]+3] - TP[joint[1] : joint[1]+3]
            angles = frame[joint[0] : joint[0]+3]
            root = PosList[i][joint[1] : joint[1]+3]
            PosList[i][joint[0] : joint[0]+3] = np.array(list(get_position(v, angles))) + root

    return PosList.reshape(-1, 15, 3)

def visualize_and_save(xyz, frames_len, filename):
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)

    fig.add_axes(ax)
    plt.ion()

    # 新增一個文本對象以顯示當前幀數
    frame_text = plt.figtext(0.5, 0.05, 'Frame 0', ha='center', fontsize=12)

    def rotation(data, alpha=0, beta=0):
        # 繞x-y軸旋轉骨架
        r_alpha = alpha * np.pi / 180
        r_beta = beta * np.pi / 180

        rx = np.array([[0, 0, 1],
                       [0, np.cos(r_alpha), -1 * np.sin(r_alpha)],
                       [0, np.sin(r_alpha), np.cos(r_alpha)]]
                      )

        ry = np.array([
            [np.cos(r_beta), 0, np.sin(r_beta)],
            [0, 0, 1],
            [-1 * np.sin(r_beta), 0, np.cos(r_beta)],
        ])

        r = ry.dot(rx)
        data = data.dot(r)
        return data

    def draw_body(frame_idx, xyz, body_color):
        all_line = []
        ax.clear()
        ax.set_axis_on()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim3d([-1, 1])
        ax.set_ylim3d([-1, 1])
        ax.set_zlim3d([-1, 1])
        ax.view_init(-90, 90)
        # ax.view_init(-0, 00)

        for part in body:
            comb_x, comb_y, comb_z = [], [], []
            for idx, joint_name in enumerate(part):
                joint_idx = joint[joint_name]
                x, y, z = xyz[frame_idx, joint_idx, :]
                comb_x.append(x)
                comb_y.append(y)
                comb_z.append(z)
            all_line.append(ax.plot(
                comb_x, comb_y, comb_z, color=body_color[0], marker='.', markerfacecolor='r')[0])
        return all_line

    body_color = 'red'
    with tqdm(total=frames_len) as t:
        def update_progress(frame_idx):
            t.update()
            frame_text.set_text(f'Frame {frame_idx}')  # 更新幀數文本
            return draw_body(frame_idx, xyz, body_color)

        anim = animation.FuncAnimation(fig, update_progress, frames=frames_len, interval=50, blit=True)
        HTML(anim.to_html5_video())
        writergif = animation.PillowWriter(fps=5)
        anim.save(filename, writer=writergif)

if __name__ == '__main__':
    i = 0
    data_dir = 'data/split_train_data/right/pkl'
    save_dir = 'train'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    while(1):
        # next_file = 'data/train_data/r_dribble_' + str(i) + '.pkl' 
        next_file = 'data/split_train_data_with_mirror_2/pkl/r_dribble_02_0.pkl'
        # if not os.path.exists(next_file):
        #     print("無檔案 => ", next_file)
        #     break

        data = np.load(next_file, allow_pickle=True)
        # data = data[660:670, :]

        print(next_file, " => 幀數 = ", np.array(data).shape[0], sep='')
        data = np.array(data).reshape(-1, 15, 3)

        # with open('data/train_data/r_dribble_' + str(i) + '.pkl', 'wb') as f:
        #     pickle.dump(data[10:, :], f)

        # 取出去掉附檔名的檔案名稱
        file_name, file_extension = os.path.splitext(next_file)

        # 製作完整的 GIF 檔案名稱
        gif_file_name = f"{file_name}.gif"
        # print("save as", gif_file_name)

        data = calculate_angle(data)
        data = calculate_position(data)
        # 存儲 GIF 檔案
        # visualize_and_save(data, data.shape[0], os.path.join(save_dir, gif_file_name))
        print(gif_file_name)
        visualize_and_save(data, data.shape[0], 'r_dribble_01_10.gif')

        break

    # for filename in os.listdir(data_dir):      
    #     # if filename.startswith('m'):
    #     #     continue
    #     if filename.endswith('.pkl'):
    #         data = np.load(os.path.join(data_dir, filename), allow_pickle=True)
    #         print(filename, " => 幀數 = ", np.array(data).shape[0], sep='')


    #         data = np.array(data).reshape(-1, 15, 3)
            
    #         # 取出去掉附檔名的檔案名稱
    #         file_name, file_extension = os.path.splitext(filename)
            
    #         # 製作完整的 GIF 檔案名稱
    #         gif_file_name = f"{file_name}.gif"
            
    #         # 存儲 GIF 檔案
    #         visualize_and_save(data, data.shape[0], os.path.join(save_dir, gif_file_name))
            