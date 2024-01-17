import os
import re
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


joint = {
    "Head": 0, "Neck": 1,
    "RightShoulder": 2, "RightArm": 3, "RightHand": 4,
    "RightHandThumb1": 5, "RightHandThumb2": 6, "RightHandThumb3": 7,
    "RightInHandIndex": 8, "RightHandIndex1": 9, "RightHandIndex2": 10, "RightHandIndex3": 11,
    "RightInHandMiddle": 12, "RightHandMiddle1": 13, "RightHandMiddle2": 14, "RightHandMiddle3": 15,
    "RightInHandRing": 16, "RightHandRing1": 17, "RightHandRing2": 18, "RightHandRing3": 19,
    "RightInHandPinky": 20, "RightHandPinky1": 21, "RightHandPinky2": 22, "RightHandPinky3": 23,
    "LeftShoulder": 24, "LeftArm": 25, "LeftHand": 26,
    "LeftHandThumb1": 27, "LeftHandThumb2": 28, "LeftHandThumb3": 29,
    "LeftInHandIndex": 30, "LeftHandIndex1": 31, "LeftHandIndex2": 32, "LeftHandIndex3": 33,
    "LeftInHandMiddle": 34, "LeftHandMiddle1": 35, "LeftHandMiddle2": 36, "LeftHandMiddle3": 37,
    "LeftInHandRing": 38, "LeftHandRing1": 39, "LeftHandRing2": 40, "LeftHandRing3": 41,
    "LeftInHandPinky": 42, "LeftHandPinky1": 43, "LeftHandPinky2": 44, "LeftHandPinky3": 45,
    "Spine": 46, "RightUpLeg": 47, "RightLeg": 48, "RightFoot": 49,
    "LeftUpLeg": 50, "LeftLeg": 51, "LeftFoot": 52,
}


spine = ['Spine', 'Neck', 'Head']
r_arm = ['Neck', 'RightShoulder', 'RightArm', 'RightHand']
r_thumb = ['RightHand', 'RightHandThumb1',
           'RightHandThumb2', 'RightHandThumb3']
r_index = ['RightHand', 'RightInHandIndex',
           'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3']
r_middle = ['RightHand', 'RightInHandMiddle',
            'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3']
r_ring = ['RightHand', 'RightInHandRing',
          'RightHandRing1', 'RightHandRing2', 'RightHandRing3']
r_pinky = ['RightHand', 'RightInHandPinky',
           'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3']

l_arm = ['Neck', 'LeftShoulder', 'LeftArm', 'LeftHand']
l_thumb = ['LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3']
l_index = ['LeftHand', 'LeftInHandIndex',
           'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3']
l_middle = ['LeftHand', 'LeftInHandMiddle',
            'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3']
l_ring = ['LeftHand', 'LeftInHandRing',
          'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3']
l_pinky = ['LeftHand', 'LeftInHandPinky',
           'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3']


r_leg = ['Spine', 'RightUpLeg', 'RightLeg', 'RightFoot']
l_leg = ['Spine', 'LeftUpLeg', 'LeftLeg', 'LeftFoot']
body = [spine, r_leg, l_leg, r_arm, r_thumb, r_index, r_middle, r_ring,
        r_pinky, l_arm, l_thumb, l_index, l_middle, l_ring, l_pinky]


def translate_pelvis_to_origin(data):
    data = np.array(data).reshape(-1, 53, 3)

    # 找到 "pelvis" 的索引，假設 "pelvis" 在第8個關節
    pelvis_index = 46

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

    return np.array(data).reshape(-1, 53)


def normalized(v):
    return v / np.linalg.norm(v)


def visualize_and_save(xyz, frames_len, filename):
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)

    fig.add_axes(ax)
    plt.ion()

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
        ax.set_xlim3d([-0.5, 0.5])
        ax.set_ylim3d([0.5, 1.5])
        ax.set_zlim3d([-3, -2])
        ax.view_init(-90, 90)
        # ax.view_init(0, 0)

        for part in body:
            comb_x, comb_y, comb_z = [], [], []
            for idx, joint_name in enumerate(part):
                joint_idx = joint[joint_name]
                x, y, z = xyz[frame_idx, joint_idx, :]
                # if joint_name.startswith("Left") or joint_name.startswith("Right"):
                #     # print("joint_name = ", joint_name)
                #     x *= 1.1
                #     y *= 1.1
                #     z *= 1.1
                comb_x.append(x)
                comb_y.append(y)
                comb_z.append(z)
            all_line.append(ax.plot(
                comb_x, comb_y, comb_z, color=body_color[0], marker='.', markerfacecolor='r', markersize=1)[0])

        return all_line

    body_color = 'red'
    with tqdm(total=frames_len) as t:
        def update_progress(frame_idx):
            t.update()
            return draw_body(frame_idx, xyz, body_color)

        anim = animation.FuncAnimation(
            fig, update_progress, frames=frames_len, interval=50, blit=True)
        HTML(anim.to_html5_video())
        writergif = animation.PillowWriter(fps=5)
        anim.save(filename, writer=writergif)


if __name__ == '__main__':
    choose_idx = np.random.randint(100, size=1).tolist()
    i = 0
    save_dir = 'img'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    while (1):
        # next_file = 'data/npy/rhand_dribble/rhand_dribble_4/' + str(i) + '.npy'
        next_file = 'l_dribble\\pkl_with_all_finger\\l_dribble_1.pkl'
        if not os.path.exists(next_file):
            print("無檔案 => ", next_file)
            break

        data = np.load(next_file, allow_pickle=True)
        print('data.shape = ', data.shape)
        data = data[:20, :]
        # data = translate_pelvis_to_origin(data)
        print("編號:", i, " => 幀數 = ", np.array(data).shape[0], sep='')

        data = np.array(data).reshape(-1, 53, 3)
        visualize_and_save(
            data, data.shape[0], save_dir+'/'+str(i+1000)+'.gif')
        i += 1
        break
