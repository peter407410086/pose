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
    def __init__(self, data_path, key_frame_path):
        """
        用於讀取運動數據集的類別。

        Args:
            data_path (str): 包含運動序列的.pkl文件的路徑。
            key_frame_path (str): 包含運動序列key_frame的.csv文件的路徑。
        """
        self.data = {}
        self.key_frames = {}
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

                    # Load the corresponding key_frame file
                    key_frame_file_path = os.path.join(key_frame_path, file_name_without_ext + '.csv')
                    if os.path.exists(key_frame_file_path):
                        # If the key_frame file exists, load the key_frame and add them to the key_frame dictionary
                        self.key_frames[file_name_without_ext] = pd.read_csv(key_frame_file_path).values
            i+=1
            if i > 300:
                break
        
        print(f"max_length = {max_length}")
        
        # Pad all sequences to the maximum length
        for key in tqdm(self.data, desc="padding sequences"):
            self.data[key] = self.pad_sequence(self.data[key], max_length)


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
        # Get the sequence and key_frames for the given file name
        data = self.data[file_name]
        key_frame = self.key_frames[file_name]
        # Return a tuple containing the sequence and key_frames
        return data, key_frame
    
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

class Key_Frame(nn.Module):
    def __init__(self, max_len=24, input_size=45, num_layers=4, hidden_size=512, num_heads=8, dropout=0.1, latent_size=256,  in_chans=3, embed_dim_ratio=32, transformer_d_model=16, joints=45):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(input_size, transformer_d_model), # [batch_size, max_len, 32] -> [batch_size, max_len, 16]
            PositionalEncoding(transformer_d_model, dropout=dropout, max_len=max_len), # [batch_size, max_len, 16] -> [batch_size, max_len, 16]
            nn.TransformerEncoder( # [batch_size, max_len, 16] -> [batch_size, max_len, 16]
                nn.TransformerEncoderLayer(transformer_d_model, num_heads, hidden_size, dropout, batch_first=True),
                num_layers
            ),
            nn.LeakyReLU(0.01), # [batch_size, max_len, 16] -> [batch_size, max_len, 16]
            nn.Linear(transformer_d_model, input_size), # [batch_size, max_len, 16] -> [batch_size, max_len, 45]
            nn.LeakyReLU(0.01), # [batch_size, max_len, 45] -> [batch_size, max_len, 45]
            nn.Flatten(), # [batch_size, max_len, 45] -> [batch_size, max_len*45]
            nn.Linear(max_len * input_size, 1) # [batch_size, max_len*45] -> [batch_size, 1]
        )

    def forward(self, x):
        print("x.shape = ", x.shape)
        x = self.encode(x)
        return torch.sigmoid(x)
    
def train(data_path, key_frame_path, max_epochs, batch_size, model_path):
    """
    Trains a KeyFramePredictor model on motion data.

    Args:
        data_path (str): The path to the motion data.
        max_epochs (int): The maximum number of training epochs.
        batch_size (int): The batch size for training.
        model_path (str): The path to save the trained model.

    Returns:
        None
    """
    dataset = MotionDataset(data_path,  key_frame_path)
    print("dataset loading finished")
    # max_len = max([dataset[i][0].shape[0] for i in range(len(dataset))])
    # print("max_len = ", max_len)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    KeyFramePredictor = Key_Frame(latent_size = latent_size, hidden_size=hidden_size)
    KeyFramePredictor = KeyFramePredictor.to('cuda')

    optimizer = optim.Adam(KeyFramePredictor.parameters(), lr=0.001, weight_decay=0.0001)

    # Reduce learning rate when a metric has stopped improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    loss_fn = nn.BCELoss()

    early_stopping = EarlyStopping(patience=10, verbose=True)

    # duration_scaler = MinMaxScaler()
    # duration_scaler = joblib.load('train/scaler/duration.pkl')

    # 在開始訓練之前，初始化兩個空列表來儲存訓練損失和驗證損失
    train_losses = []
    val_losses = []
    for epoch in range(max_epochs):
        KeyFramePredictor.train()
        train_loss = 0.0
        for i, (x, key_frame) in enumerate(tqdm(train_loader, desc="training")):
            x = x.float().to(device)
            key_frame = key_frame.to(device)

            optimizer.zero_grad()
            x_hat = KeyFramePredictor(x)
            print("x_hat[0] = ", x_hat[0])
            print("key_frame[0] = ", key_frame[0])
            print(f"x_hat.shape = {x_hat.shape}, key_frame.shape = {key_frame.shape}")
            loss = loss_fn(x_hat, x)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        # print(f"Epoch {epoch+1}/{max_epochs}, train_loss: {train_loss:.4f}")

        # Calculate validation loss
        KeyFramePredictor.eval()
        val_loss = 0.0
        # with torch.no_grad():
        #     for i, (x, label, key_frame) in enumerate(tqdm(val_loader, desc="validation")):
        #         x = x.float().to(device)
        #         label = label.float().to(device)

        #         optimizer.zero_grad()
        #         x_hat, mu, logvar = KeyFramePredictor(x, label)

        #         loss = loss_fn(x_hat, x)

        #         kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #         loss += kl_loss
        #         val_loss += loss.item()

        #     val_loss /= len(val_loader)

        #     # adjust learning rate if needed (early stopping)
        #     scheduler.step(val_loss)
            
        #     # get current learning rate
        #     current_lr = optimizer.param_groups[0]['lr']

        #     # print learning rate and loss for this epoch
        #     print(f"Epoch {epoch+1}/{max_epochs}, train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, lr: {current_lr:.6f}")

        #     # Call early stopping
        #     early_stopping(val_loss, KeyFramePredictor)

        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         break
        
        # 在每個 epoch 結束時，將訓練損失和驗證損失添加到列表中
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    # 在所有 epoch 結束後，繪製折線圖
    plt.plot(np.log(np.log(train_losses)), label='Train Loss')
    plt.plot(np.log(np.log(val_losses)), label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()    # 顯示圖例
    plt.grid(True)  # 顯示網格
    plt.savefig('result/loss_plot.jpg') # 儲存圖片

    torch.save(KeyFramePredictor.state_dict(), model_path)

if __name__ == "__main__":
    data_path = "data/split_train_data_with_mirror_2/angle_pkl"
    key_frame_path = "data/split_train_data_with_mirror_2/key_frame"
    model_path = "train/pth/split_train_data_with_mirror_key_frame.pth"
    # data_path = 'test/pkl'
    # model_path = "train/pth/11.pth"
    max_epochs = 300
    batch_size = 64
    num_layers = 4
    hidden_size = 512
    dropout = 0.1
    latent_size = 512
    input_size = 45

    # move model to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # train model and save model as pth file
    train(data_path, key_frame_path, max_epochs, batch_size, model_path)

    # save_name = "poseformer_angle.gif"

    # generate sample data and save as pkl and gif file
    # generate(num_samples=20,  model_path=model_path)
