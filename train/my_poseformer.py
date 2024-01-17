import os
import math
import joblib
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
from IPython.core.display import HTML
from matplotlib import animation
from pykalman import KalmanFilter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

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

spine = ['pelvis', 'neck', 'head']
r_arm = ['neck', 'rshoulder', 'rarm', 'rhand']
l_arm = ['neck', 'lshoulder', 'larm', 'lhand']
r_leg = ['pelvis', 'rthigh', 'rknee', 'rankle']
l_leg = ['pelvis', 'lthigh', 'lknee', 'lankle']
body  = [spine, r_leg, l_leg, r_arm, l_arm]

# def calculate_mpjpe(x_hat):
#     """
#     Calculate the mean per joint position error (MPJPE) for a sequence of poses.
#     """  
#     mpjpe = 0.0
    
#     # Get the number of frames in the sequence
#     frames = x_hat.shape[0]

#     # Get the ground truth poses for the sequence
#     ground_truth = np.load('data/split_train_data_with_mirror_2/angle_pkl/r_dribble_02_0.pkl', allow_pickle=True)
#     ground_truth = calculate_position(ground_truth)

#     frame_mpjpe = np.zeros(frames)
#     for i in range(frames):
#         # Calculate the MPJPE for the current frame
#         mpjpe_frame = np.linalg.norm(ground_truth[i] - x_hat[i], axis=0)
#         frame_mpjpe[i] = np.mean(mpjpe_frame)

#     # Calculate the average MPJPE across all frames
#     avg_mpjpe = float(np.mean(frame_mpjpe)) * 1000.0 # m -> mm 
#     print(f"Final average MPJPE = {avg_mpjpe}")

#     # Return the average MPJPE across all frames
#     return avg_mpjpe 

def calculate_mpjpe(x_hat):
    """
    Calculate the mean per joint position error (MPJPE) for a sequence of poses.
    """  
    mpjpe = 0.0
    
    # Get the number of frames in the sequence
    frames = x_hat.shape[0]

    # Get the ground truth poses for the sequence
    ground_truth = np.load('data/split_train_data_with_mirror_2/angle_pkl/r_dribble_02_0.pkl', allow_pickle=True)
    ground_truth = calculate_position(ground_truth)
    ground_truth = ground_truth.reshape(frames, 15, 3)
    x_hat = x_hat.reshape(frames, 15, 3)

    frame_mpjpe = np.zeros(frames)
    for i in range(frames):
        # Calculate the MPJPE for the current frame
        mpjpe_frame = np.linalg.norm(ground_truth[i] - x_hat[i], axis=1)
        frame_mpjpe[i] = np.mean(mpjpe_frame)

    # Calculate the average MPJPE across all frames
    avg_mpjpe = float(np.mean(frame_mpjpe)) * 1000.0 # m -> mm 
    print(f"Final average MPJPE = {avg_mpjpe}")

    # Return the average MPJPE across all frames
    return avg_mpjpe

def calculate_hand_mpjpe(x_hat):
    """
    Calculate the mean hand position error (MPJPE) for a sequence of poses.
    """
    mpjpe = 0.0
    # Get the number of frames in the sequence
    frames = x_hat.shape[0]
    # print(f"frames = {frames}")

    # Get the ground truth poses for the sequence
    ground_truth = np.load('data/split_train_data_with_mirror_2/angle_pkl/r_dribble_01_10.pkl', allow_pickle=True)
    ground_truth = calculate_position(ground_truth) # [frames, 45]

    for i in range(frames-1):
        # Calculate the MPJPE for the current frame
        mpjpe += np.mean(np.linalg.norm(ground_truth[i][12:15] - x_hat[i][12:15], axis=0))

    # Return the average MPJPE across all frames
    return mpjpe * 100 / frames

def kalman_1D(observations, damping=1.0):
    """
    Apply a 1D Kalman filter to a sequence of observations.

    Parameters:
        observations (list or numpy array): The sequence of observations.
        damping (float): The damping factor for the Kalman filter.

    Returns:
        numpy array: The smoothed time series data.
    """
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1

    kf = KalmanFilter(
        initial_state_mean=initial_value_guess,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrix
    )
    pred_state, _ = kf.smooth(observations)
    return pred_state

def kalman_filter(data):
    """
    Apply the Kalman filter to a matrix of data.

    Parameters:
        data (numpy array): A matrix of time series data.

    Returns:
        numpy array: The smoothed data after applying the Kalman filter.
    """
    kalman = [kalman_1D(joints, damping=0.01) for joints in data.T]
    kalman = np.array(kalman).T[0]
    return kalman

def calculate_max_len(data_path, label_path):
    dataset = MotionDataset(data_path, label_path)
    max_len = max([dataset[i][0].shape[0] for i in range(len(dataset))])
    return max_len

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

def get_position(v, angles):
        r = np.linalg.norm(v)
        x = r*angles[0]
        y = r*angles[1]
        z = r*angles[2]
        return  x,y,z

def calculate_position(fullbody):
    TP_data = np.load('data/split_train_data_with_mirror_2/pkl/r_dribble_01_10.pkl', allow_pickle=True)
    TP = TP_data[0]
    PosList = np.zeros_like(fullbody)
    for i, frame in enumerate(fullbody):
        for joint in jointConnect:
            v = TP[joint[0] : joint[0]+3] - TP[joint[1] : joint[1]+3]
            angles = frame[joint[0] : joint[0]+3]
            root = PosList[i][joint[1] : joint[1]+3]
            PosList[i][joint[0] : joint[0]+3] = np.array(list(get_position(v, angles))) + root
    return PosList

def visualize_and_save(xyz, frames_len, filename):
    xyz = xyz.reshape(frames_len, 15, 3)
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
        # ax.set_axis_off()  # 隱藏座標軸

        # 找到 "pelvis" 的索引
        pelvis_index = joints['pelvis']

        # 獲取 "pelvis" 的座標
        center_x = np.mean(xyz[pelvis_index, 0])
        center_y = np.mean(xyz[pelvis_index, 1])
        center_z = np.mean(xyz[pelvis_index, 2])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 將軸的限制設置為以 "pelvis" 的座標為中心的範圍
        ax.set_xlim3d((center_x - 1.0, center_x + 1.0))
        ax.set_ylim3d((center_y - 1.0, center_y + 1.0))
        ax.set_zlim3d((center_z - 1.0, center_z + 1.0))

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_xlim3d((-1.0, 1.0))
        # ax.set_ylim3d((-1.0, 1.0))
        # ax.set_zlim3d((-1.0, 1.0))

        ax.view_init(-90, 90)
        # ax.view_init(-0, 00)

        for part in body:
            comb_x, comb_y, comb_z = [], [], []
            for idx, joint_name in enumerate(part):
                joint_idx = joints[joint_name]
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

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), model_path)
        self.val_loss_min = val_loss

class MotionDataset(Dataset):
    def __init__(self, data_path, label_path):
        """
        用於讀取運動數據集的類別。

        Args:
            data_path (str): 包含運動序列的.pkl文件的路徑。
            label_path (str): 包含運動序列標籤的.csv文件的路徑。
        """
        self.data = {}
        self.labels = {}
        max_length = 0
        i = 0

        # Loop through all files in the data_path directory
        for filename in tqdm(os.listdir(data_path), desc="loading data"):
            if filename.endswith('.pkl'):
                file_path = os.path.join(data_path, filename)
                with open(file_path, 'rb') as f:
                    # Remove the file extension (.pkl) from the filename
                    file_name_without_ext = os.path.splitext(filename)[0]
                    # Load the sequence from the pickle file
                    sequence = pickle.load(f)
                    # Update the maximum length if this sequence is longer
                    max_length = max(max_length, sequence.shape[0])
                    # Add the sequence to the data dictionary
                    self.data[file_name_without_ext] = sequence

                    # Load the corresponding label file
                    label_file_path = os.path.join(label_path, file_name_without_ext + '.csv')
                    if os.path.exists(label_file_path):
                        # If the label file exists, load the labels and add them to the labels dictionary
                        self.labels[file_name_without_ext] = pd.read_csv(label_file_path).values                       
            # if i == 1:
            #     break
        
        print(f"max_length = {max_length}")
        
        # Pad all sequences to the maximum length
        for key in tqdm(self.data, desc="padding sequences"):
            self.data[key] = self.pad_sequence(self.data[key], max_length)

        # # 選擇要標準化的特徵的列索引，dribbling_frequency,bending_angle,duration需要標準化
        # features_indices = range(4, 7)

        # # 對每一列的數據進行標準化
        # for feature_index in tqdm(features_indices, desc="normalizing features"):
        #     # 決定要標準化的特徵名稱
        #     label_name = ""
        #     if feature_index == 4:
        #         label_name = "dribbling_frequency"
        #     elif feature_index == 5:
        #         label_name = "bending_angle"
        #     elif feature_index == 6:
        #         label_name = "duration"
        #     # 創建一個列表來儲存該列的所有數據
        #     column_data = []

        #     # print(f"feature_index = {feature_index}, label_name = {label_name}")

        #     for key in self.labels.keys():
        #         column_data.append(self.labels[key][0][feature_index])

        #     # Convert the list to a numpy array
        #     column_data = np.array(column_data).reshape(-1, 1)
        #     # print(f"column_data = {column_data}")

        #     # Create a StandardScaler
        #     scaler = MinMaxScaler()

        #     # Train the StandardScaler using all the data in the column
        #     scaler.fit(column_data)

        #     # print(f"scaler.data_max_ = {scaler.data_max_}")
        #     # print(f"scaler.data_min_ = {scaler.data_min_}")

        #     # joblib.dump(scaler, f'train/scaler/{label_name}.pkl')

        #     # Standardize each data point in the column
        #     for key in self.labels.keys():
        #         self.labels[key][0][feature_index] = scaler.transform(np.array(self.labels[key][0][feature_index]).reshape(-1, 1))[0][0]
        #         # print(f"self.labels[{key}][0][{feature_index}] = {self.labels[key][0][feature_index]}")

        # 印出所有標準化後的數據
        # print("標準化後的數據")
        # for key in self.labels.keys():
        #     print(f"self.labels[{key}] = {self.labels[key]}")

        # Store the file names in a list
        self.file_names = list(self.data.keys())

    def __len__(self):
        """
        返回數據集的大小。

        Returns:
            int: 數據集的大小。
        """
        # Return the number of files in the dataset
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        根據索引返回數據集中的一個樣本。

        Args:
            idx (int): 樣本的索引。

        Returns:
            tuple: 包含運動序列和標籤的元組。
        """
        # Get the file name for the given index
        file_name = self.file_names[idx]
        # Get the sequence and label for the given file name
        data = self.data[file_name]
        label = self.labels[file_name]
        # Return a tuple containing the sequence and label
        return data, label
    
    def pad_sequence(self, sequence, max_length):
        """
        將序列填充到最大長度。

        Args:
            sequence (numpy.ndarray): 要填充的序列。
            max_length (int): 最大長度。

        Returns:
            numpy.ndarray: 填充後的序列。
        """
        # Get the length of the sequence
        sequence_length = sequence.shape[0]
        # Calculate the amount of padding needed
        padding_length = max_length - sequence_length
        if padding_length > 0:
            # If the sequence length is less than the maximum length, pad the sequence with two
            sequence = np.pad(sequence, ((0, padding_length), (0, 0)), 'constant', constant_values=0)
        else:
            # If the sequence length is greater than the maximum length, truncate the sequence
            sequence = sequence[:max_length]
        # Return the padded or truncated sequence
        return sequence

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CVAE(nn.Module):
    def __init__(self, max_len=24, input_size=45, num_layers=4, transformer_hidden_size=512, num_heads=8, dropout=0.1, latent_size=256, label_size=7, in_chans=3, embed_dim_ratio=32, model_size=16, joints=15):
        super().__init__()

        self.spatial_encoder = nn.Sequential(
            nn.Linear(in_chans, embed_dim_ratio), # [batch_size, max_len, 3] -> [batch_size, max_len, 32]
            PositionalEncoding(embed_dim_ratio, dropout=dropout, max_len=joints), # [batch_size, max_len, 32]
            nn.TransformerEncoder( # [batch_size, max_len, 32] -> [batch_size, max_len, 32]
                nn.TransformerEncoderLayer(embed_dim_ratio, num_heads, transformer_hidden_size, dropout, batch_first=True),
                num_layers
            )
        )
        
        self.temporal_encoder = nn.Sequential(
            nn.Linear(embed_dim_ratio*joints, model_size), # [batch_size, max_len, 32*15] -> [batch_size, max_len, 16
            PositionalEncoding(model_size, dropout=dropout, max_len=max_len), # [batch_size, max_len, 16] -> [batch_size, max_len, 16]
            nn.TransformerEncoder( # [batch_size, max_len, 16] -> [batch_size, max_len, 16]
                nn.TransformerEncoderLayer(model_size, num_heads, transformer_hidden_size, dropout, batch_first=True),
                num_layers
            ),
            nn.LeakyReLU(0.01), # [batch_size, max_len, 16]
            nn.Linear(model_size, input_size), # [batch_size, max_len, 16] -> [batch_size, max_len, 45]
            nn.LeakyReLU(0.01), # [batch_size, max_len, 45]
            nn.Flatten(), # [batch_size, max_len, 45] -> [batch_size, max_len*45]
            nn.Linear(max_len * input_size, latent_size*2) # [batch_size, max_len*45] -> [batch_size, latent_size*2]
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size + label_size, max_len * input_size), # [batch_size, latent_size + label_size] -> [batch_size, max_len*45]
            # reshape to [batch_size, max_len, input_size]
            nn.Unflatten(1, (max_len, input_size)), # [batch_size, max_len*45] -> [batch_size, max_len, 45]
            nn.LeakyReLU(0.01), # [batch_size, max_len, input_size]
            nn.Linear(input_size, model_size), # [batch_size, max_len, input_size] -> [batch_size, max_len, model_size]
            nn.LeakyReLU(0.01), # [batch_size, max_len, model_size]
            nn.TransformerEncoder( # [batch_size, max_len, model_size]
                nn.TransformerEncoderLayer(model_size, num_heads, transformer_hidden_size, dropout, batch_first=True),
                num_layers
            ),
            nn.Linear(model_size, input_size) # [batch_size, max_len, model_size] -> [batch_size, max_len, input_size]
        )

    def encode(self, x): # x:[batch_size, frames, 45]
        # b is batch size, f is number of frames, j is number of joints, c is xyz coordinates
        # reshape x from (b, f, j*c) to (b, f, j, c)
        x = rearrange(x, 'b f (j c) -> b f j c', j = 15, c = 3)

        b, f, j, c = x.shape  

        # reshape x from (b, f, j, c) to (b*f, j, c)
        x = rearrange(x, 'b f j c -> (b f) j c', )

        # pass x through the spatial encoder
        x = self.spatial_encoder(x) # [batch_size*frames, joints, 3] -> [batch_size*frames, joints, 32]

        # reshape x from (b*f, j, e) to (b, f, j, e) , e is embed_dim_ratio
        x = rearrange(x, '(b f) j e -> b f (j e)', b = b, f = f) # [batch_size*frames, joints, 32] -> [batch_size, frames, joints*32]

        # pass x through the temporal encoder
        x = self.temporal_encoder(x) # [batch_size, frames, joints*32] -> [batch_size, latent_size*2]

        # split the encoded tensor into mu and logvar
        mu, logvar = torch.chunk(x, 2, dim=1) # [batch_size, latent_size*2] -> [batch_size, latent_size], [batch_size, latent_size]

        # return mu and logvar tensors
        return mu, logvar

    def decode(self, z, label):
        label = label.squeeze(dim=1) # [batch_size, 1, label_size] -> [batch_size, label_size]
        z = torch.cat([z, label], dim=1) # [batch_size, latent_size] + [batch_size, label_size] -> [batch_size, latent_size + label_size]
        x = self.decoder(z) # [batch_size, latent_size + label_size] -> [batch_size, frames, 45]
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, label): # x: [batch_size, frames, 45]
        mu, logvar = self.encode(x)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
            
        x_hat = self.decode(z, label)

        duration_scaler = MinMaxScaler()
        duration_scaler = joblib.load('train/scaler/duration.pkl')

        # 從 label 中獲取持續時間
        duration = label[:, :, 6]
        # print(f"duration[0] = {duration[0].float().item():.2f}")

        # # 將 duration 移到 CPU，然後轉換為 numpy 陣列並重塑
        # duration_2d = duration.cpu().numpy().reshape(-1, 1)

        # # 使用 inverse_transform 進行反向轉換
        # original_duration = duration_scaler.inverse_transform(duration_2d)

        # # 將結果轉換回原始的形狀
        # original_duration = original_duration.reshape(duration.shape)

        # # 將 numpy 陣列轉換為 PyTorch 張量，並確保它在正確的設備上
        # duration = torch.from_numpy(original_duration).to(x_hat.device)
        # # print(f"after duration[0] = {duration[0].float().item():.2f}")

        # 創建一個與 x_hat 相同形狀的零張量
        zeros = torch.zeros_like(x_hat)

        # 創建一個遮罩，其中的值為 True 如果當前時間步長小於持續時間，否則為 False
        mask = torch.arange(x_hat.size(1), device=x_hat.device).expand(x_hat.size(0), x_hat.size(1)) < duration.view(-1, 1)

        # 使用遮罩來替換 x_hat，如果當前時間步長大於或等於持續時間，則 x_hat 的值為 (0,0,0)
        x_hat = torch.where(mask.unsqueeze(-1), x_hat, zeros)

        return x_hat, mu, logvar

def train(data_path, label_path, max_epochs, batch_size, model_path):
    """
    Trains a CVAE model on motion data.

    Args:
        data_path (str): The path to the motion data.
        label_path (str): The path to the label data.
        max_epochs (int): The maximum number of training epochs.
        batch_size (int): The batch size for training.
        model_path (str): The path to save the trained model.

    Returns:
        None
    """
    dataset = MotionDataset(data_path, label_path)
    print("dataset loading finished")

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    cvae = CVAE(label_size = dataset[0][1].shape[1], latent_size = latent_size, transformer_hidden_size=transformer_hidden_size)
    cvae = cvae.to('cuda')

    optimizer = optim.Adam(cvae.parameters(), lr=0.0001, weight_decay=0.0001)

    # Reduce learning rate when a metric has stopped improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    loss_fn = nn.MSELoss()

    early_stopping = EarlyStopping(patience=10, verbose=True)

    duration_scaler = MinMaxScaler()
    duration_scaler = joblib.load('train/scaler/duration.pkl')

    # 在開始訓練之前，初始化兩個空列表來儲存訓練損失和驗證損失
    train_losses = []
    val_losses = []
    for epoch in range(max_epochs):
        cvae.train()
        train_loss = 0.0
        for i, (x, label) in enumerate(tqdm(train_loader, desc="training")):
            # # Create a tensor of zeros with shape 
            # mask = torch.zeros(x.size(0), x.size(1), 45, device=device) # [batch_size, max_len, 45]
            # for j in range(x.size(0)):
            #     # Get the sequence length for the j-th sample
            #     n = int(label[j][0][6])
            #     # n = int(duration_scaler.inverse_transform(np.array(label[j][0][6]).reshape(-1, 1))[0][0])
                
            #     # print("Duration for sample", j, ":", n)
            #     # print("Values after duration for sample", j, ":", x[j, :, 0])
            #     # Set the first n elements of the j-th sample in the mask tensor to 1
            #     mask[j, :n, :] = 1

            x = x.float().to(device)
            label = label.float().to(device)

            optimizer.zero_grad()
            x_hat, mu, logvar = cvae(x, label)

            # n = int(label[i][0][6])
            # print(f"x_hat[0][{n-1}][0:3] = {x_hat[0][n-1][0:3]}")
            # print(f"x_hat[0][{n}][0:3] = {x_hat[0][n:][0:3]}")

            # # Create a boolean mask where padding positions are False and non-padding positions are True
            # bool_mask = (mask == 1)

            # # Apply the boolean mask to x_hat and x
            # x_hat_masked = x_hat[bool_mask]
            # x_masked = x[bool_mask]

            # # 試看看需不需要 mask
            # loss = loss_fn(x_hat_masked, x_masked)

            loss = loss_fn(x_hat, x)

            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss += kl_loss
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        # print(f"Epoch {epoch+1}/{max_epochs}, train_loss: {train_loss:.4f}")

        # Calculate validation loss
        cvae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (x, label) in enumerate(tqdm(val_loader, desc="validation")):
                # # Create a tensor of zeros with shape 
                # mask = torch.zeros(x.size(0), x.size(1), 45, device=device) # [batch_size, max_len, 45]
                # for i in range(x.size(0)):
                #     # Get the sequence length for the i-th sample
                #     n = int(label[i][0][6])
                #     # n = int(duration_scaler.inverse_transform(np.array(label[i][0][6]).reshape(-1, 1))[0][0])

                #     # Set the first n elements of the i-th sample in the mask tensor to 1
                #     mask[i, :n, :] = 1

                x = x.float().to(device)
                label = label.float().to(device)

                optimizer.zero_grad()
                x_hat, mu, logvar = cvae(x, label)

                # # Create a boolean mask where padding positions are False and non-padding positions are True
                # bool_mask = (mask == 1)

                # # Apply the boolean mask to x_hat and x
                # x_hat_masked = x_hat[bool_mask]
                # x_masked = x[bool_mask]

                # # 試看看需不需要 mask
                # loss = loss_fn(x_hat_masked, x_masked)
                loss = loss_fn(x_hat, x)

                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss += kl_loss
                val_loss += loss.item()

            val_loss /= len(val_loader)

            # adjust learning rate if needed (early stopping)
            scheduler.step(val_loss)
            
            # get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # print learning rate and loss for this epoch
            print(f"Epoch {epoch+1}/{max_epochs}, train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, lr: {current_lr:.6f}")

            # Call early stopping
            early_stopping(val_loss, cvae)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        # 在每個 epoch 結束時，將訓練損失和驗證損失添加到列表中
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    # 在所有 epoch 結束後，繪製折線圖
    plt.plot(np.log(train_losses), label='Train Loss')
    plt.plot(np.log(val_losses), label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend() # 顯示圖例
    plt.grid(True) # 顯示網格
    plt.savefig('result/loss_plot.jpg')  # 儲存圖片

    torch.save(cvae.state_dict(), model_path)

def generate(num_samples, label_path, model_path):
    def normalize_label(label):
        # Create a min-max scaler object
        dribbling_frequency_scaler = MinMaxScaler()
        bending_angle_scaler = MinMaxScaler()
        duration_scaler = MinMaxScaler()

        # Load scaler
        dribbling_frequency_scaler = joblib.load('train/scaler/dribbling_frequency.pkl')
        bending_angle_scaler = joblib.load('train/scaler/bending_angle.pkl')
        duration_scaler = joblib.load('train/scaler/duration.pkl')

        print(f"label = {label}")
        
        # Normalize the label data
        label[0][4] = dribbling_frequency_scaler.transform(np.array(label[0][4]).reshape(-1, 1))[0][0]
        label[0][5] = bending_angle_scaler.transform(np.array(label[0][5]).reshape(-1, 1))[0][0]
        # label[0][6] = duration_scaler.transform(np.array(label[0][6]).reshape(-1, 1))[0][0]

        print(f"label = {label}")
        return label

    torch.manual_seed(0)

    cvae = CVAE(label_size = label_size, latent_size = latent_size, transformer_hidden_size = transformer_hidden_size)
    cvae.load_state_dict(torch.load(model_path))
    cvae = cvae.to(device)
    cvae.eval()

    # 初始化一個空的列表來儲存 MPJPE 的結果
    mpjpe_results = []
    hand_mpjpe_results = []

    with torch.no_grad():
        for i in range(num_samples):
            mean = torch.tensor([0]).to(device)  # 固定平均值
            variances = [1 for i in range(num_samples)]  # 可變變異數
            # z = torch.randn(1, latent_size).to(device) * torch.sqrt(torch.tensor([variances[i % len(variances)]]).to(device)) + mean
            # z = torch.randn(1, latent_size).to(device)
            z = 2 * torch.rand(1, latent_size).to(device) - 1
            # print(f"z mean: {z.mean():.2f}, z variance: {z.var():.2f}")

            label_file_path = "train/r_dribble_01_10.csv" #####
            # label_file_path = "test/csv/2.csv" #####
            if os.path.exists(label_file_path):
                label = pd.read_csv(label_file_path).values
                # print(f"label.type = {type(label)}")
                frame_len = int(label[0][6])
                # label = normalize_label(label)
                label = torch.tensor(label).float().to(device)

                x_hat = cvae.decode(z, label)
                save_path = f"result/generated_{i+1}.gif"
                data = np.array(x_hat.squeeze().tolist())
                
                data = calculate_position(data)
                data = data[:frame_len,:]
                # print(f"data = {data}")

                # calculate_hand_mpjpe(data)

                if i == 0:
                    visualize_and_save(data, frame_len, save_path)

                mpjpe = calculate_mpjpe(data)
                hand_mpjpe = calculate_hand_mpjpe(data)
                # print(f"mpjpe: {mpjpe:.2f}")

                # 將結果儲存到列表中
                mpjpe_results.append(mpjpe)
                hand_mpjpe_results.append(hand_mpjpe)

            else:
                print(f"Label file for sample {i+1} not found.")

    # 畫出 MPJPE 的結果
    plt.figure()
    bp = plt.boxplot(mpjpe_results, showfliers=False)  # 不顯示異常值
    for median in bp['medians']:
        median.set(color='orange', linewidth=2, label='Median')  # 設定中位數線的顏色、線寬和標籤
    plt.plot(1, float(np.mean(mpjpe_results)), 'rD', label='Mean')  # 繪製平均值並添加標籤 # 'rD' 是一個格式字符串，表示點的顏色和形狀。在這裡，'r' 表示紅色，'D' 表示鑽石形狀。
    plt.scatter(np.array([0.75]*len(mpjpe_results)), mpjpe_results)  # 繪製數據點
    plt.text(1.1, float(np.mean(mpjpe_results)), f"Mean: {np.mean(mpjpe_results):.2f}", va='center', ha='left')  # 添加平均值標籤
    plt.legend()  # 顯示圖例
    
    plt.xlabel('per joint')
    plt.ylabel('MPJPE (mm)')
    plt.savefig('result/mpjpe_results.jpg')

    # 畫出 HAND MPJPE 的結果
    plt.figure()
    bp = plt.boxplot(hand_mpjpe_results, showfliers=False)  # 不顯示異常值
    for median in bp['medians']:
        median.set(color='orange', linewidth=2, label='Median')  # 設定中位數線的顏色、線寬和標籤
    plt.plot(1, float(np.mean(hand_mpjpe_results)), 'rD', label='Mean')  # 繪製平均值並添加標籤 # 'rD' 是一個格式字符串，表示點的顏色和形狀。在這裡，'r' 表示紅色，'D' 表示鑽石形狀。
    plt.scatter(np.array([0.75]*len(hand_mpjpe_results)), hand_mpjpe_results)  # 繪製數據點
    plt.text(1.1, float(np.mean(hand_mpjpe_results)), f"Mean: {np.mean(hand_mpjpe_results):.2f}", va='center', ha='left')  # 添加平均值標籤
    plt.legend()  # 顯示圖例
    
    plt.xlabel('Hand')
    plt.ylabel('MPJPE (cm)')
    plt.savefig('result/hand_mpjpe_results.jpg')

    # 畫出 MPJPE 的結果(var增加，mpjpe是否也會變大)
    # plt.figure()
    # plt.plot(mpjpe_results)
    # plt.xlabel('Variance')
    # variances = [0.5 + i * 0.25 for i in range(11)]
    # plt.xticks(range(len(variances)), [str(v) for v in variances])  # Fix: Convert variances to strings
    # plt.ylabel('MPJPE (cm)')

if __name__ == "__main__":
    data_path = "data/split_train_data_with_mirror_2/angle_pkl"
    label_path = "data/split_train_data_with_mirror_2/csv"
    model_path = "train/pth/split_train_data_with_mirror.pth"
    # data_path = 'test/pkl'
    # label_path = 'test/csv'
    # model_path = "train/pth/11.pth"
    max_epochs = 300
    batch_size = 64
    num_layers = 4
    dropout = 0.1
    latent_size = 512
    transformer_hidden_size = 512

    input_size = 45
    label_size = 7

    # move model to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # calculate max_len
    # max_len = calculate_max_len(data_path, label_path)
    # max_len = 24

    # train model and save model as pth file
    # train(data_path, label_path, max_epochs, batch_size, model_path)

    # generate sample data and save as pkl and gif file
    generate(num_samples = 1, label_path = label_path, model_path = model_path)