import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
from torch import optim
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

joint = {"head": 0, "neck": 1, "rshoulder": 2, "rarm": 3, "rhand": 4,
         "lshoulder": 5, "larm": 6, "lhand": 7, "pelvis": 8,
         "rthing": 9, "rknee": 10, "rankle": 11, "lthing": 12, "lknee": 13, "lankle": 14}

spine = ['pelvis', 'neck', 'head']
r_arm = ['neck', 'rshoulder', 'rarm', 'rhand']
l_arm = ['neck', 'lshoulder', 'larm', 'lhand']
r_leg = ['pelvis', 'rthing', 'rknee', 'rankle']
l_leg = ['pelvis', 'lthing', 'lknee', 'lankle']
body = [spine, r_leg, l_leg, r_arm, l_arm]


def normalized(v):
    return v / np.linalg.norm(v)

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

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
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
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len, feature_size = x.size()
        # [batch_size = 128, seq_len = 30]
        # print(f"batch_size: {batch_size}, seq_len: {seq_len}, feature_size: {feature_size}")
        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]

class CVAE(nn.Module):
    def __init__(self, input_size, label_size, output_size, num_layers, hidden_size, num_heads, dropout, latent_size, max_len, device):
        super().__init__()
        pe_size = input_size if input_size % 2 == 0 else input_size + 1

        self.encoder = nn.Sequential(
            nn.Linear(input_size+pe_size, input_size),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout, batch_first=True),
                num_layers
            ),
            nn.LeakyReLU(0.01),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(0.01),
            nn.Flatten(),
            nn.Linear(max_len * input_size, latent_size*2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + label_size, max_len * input_size),
            # reshape to [batch_size, max_len, input_size]
            nn.Unflatten(1, (max_len, input_size)),
            nn.LeakyReLU(0.01),
            nn.Linear(input_size, input_size),
            # nn.TransformerDecoder(
            #     nn.TransformerDecoderLayer(input_size, num_heads, hidden_size, dropout, batch_first=True),
            #     num_layers
            # ),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout, batch_first=True),
                num_layers
            ),
            nn.Linear(input_size, input_size)
        )

        self.positional_encoding = PositionalEncoding(pe_size, max_len, device=device)

    def encode(self, x):
        pe = self.positional_encoding(x)
        # extend pe to batch_size
        pe = pe.expand(x.shape[0], -1, -1)
        # print(f"pos_encode_x: {pe.shape}")
        x = torch.cat([x, pe], dim=2)

        mu, logvar = torch.chunk(self.encoder(x), 2, dim=1)
        # print(f"mu: {mu.shape}, logvar: {logvar.shape}")
        return mu, logvar

    def decode(self, z, label):
        #z: torch.Size([32, 128]), label: torch.Size([32, 1, 7])
        # print(f"z: {z.shape}, label: {label.shape}")
        label = label.squeeze(dim=1)
        z = torch.cat([z, label], dim=1)
        # print(f"cat_z: {z.shape}")
        x = self.decoder(z)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, label):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, label)

        # 從 label 中獲取持續時間
        duration = label[:, :, 6].long()

        # 創建一個與 x_hat 相同形狀的零張量
        zeros = torch.zeros_like(x_hat)

        # 創建一個遮罩，其中的值為 True 如果當前時間步長小於持續時間，否則為 False
        mask = torch.arange(x_hat.size(1), device=x_hat.device).expand(x_hat.size(0), x_hat.size(1)) < duration.view(-1, 1)

        # 使用遮罩來替換 x_hat，如果當前時間步長大於或等於持續時間，則 x_hat 的值為 (0,0,0)
        x_hat = torch.where(mask.unsqueeze(-1), x_hat, zeros)

        return x_hat, mu, logvar

class MotionDataset(Dataset):
    def __init__(self, data_path, label_path):
        """
        用於讀取運動數據集的類別。

        Args:
            data_path (str): 包含運動序列的.pkl文件的路徑。
            label_path (str): 包含運動序列標籤的.csv文件的路徑。
        """
        # Initialize the MotionDataset object
        self.data = {}
        self.labels = {}
        max_length = 0
        i = 0
        # Loop through all files in the data_path directory
        for filename in os.listdir(data_path):
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
            # if i == 3:
            #     break
        # Pad all sequences to the maximum length
        for key in self.data:
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

def train(data_path, label_path, max_epochs, batch_size, num_layers, hidden_size, num_heads, dropout, latent_size, save_path):
    dataset = MotionDataset(data_path, label_path)
    print("dataset loading finished")
    max_len = max([dataset[i][0].shape[0] for i in range(len(dataset))])
    
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    cvae = CVAE(input_size=dataset[0][0].shape[1], label_size = dataset[0][1].shape[1], output_size=dataset[0][0].shape[1], num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads, dropout=dropout, latent_size=latent_size, max_len = max_len, device=device)
    cvae = cvae.to('cuda')

    optimizer = optim.Adam(cvae.parameters(), lr=0.001, weight_decay=0.0001)
    # 定義學習率調度器，如果在 5 個 epochs 內驗證損失沒有改進，則學習率乘以 0.1
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    loss_fn = nn.MSELoss()

    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in range(max_epochs):
        cvae.train()
        train_loss = 0.0
        for i, (x, label) in enumerate(tqdm(train_loader)):
            x = x.float().to(device)
            label = label.float().to(device)
            # print("Shape of x:", x.shape)
            # print("Shape of label:", label.shape)

            # Create a tensor of zeros with shape [batch_size, max_len, 45]
            # mask = torch.zeros(x.size(0), x.size(1), 45, device=device)
            # for i in range(x.size(0)):
            #     # Get the sequence length for the i-th sample
            #     n = int(label[i][0][6])
            #     print("Duration for sample", i, ":", n)
            #     print("Values after duration for sample", i, ":", x[i, n:, 0])
            #     # Set the first n elements of the i-th sample in the mask tensor to 1
            #     mask[i, :n, :] = 1

            optimizer.zero_grad()
            x_hat, mu, logvar = cvae(x, label)

            # Create a boolean mask where padding positions are False and non-padding positions are True
            # bool_mask = (mask == 1)

            # Apply the boolean mask to x_hat and x
            # x_hat_masked = x_hat[bool_mask]
            # x_masked = x[bool_mask]

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
            for i, (x, label) in enumerate(tqdm(val_loader)):
                x = x.float().to(device)
                label = label.float().to(device)

                # Create a tensor of zeros with shape [batch_size, max_len, 45]
                # mask = torch.zeros(x.size(0), x.size(1), 45, device=device)
                # for i in range(x.size(0)):
                #     # Get the sequence length for the i-th sample
                #     n = int(label[i][0][6])
                #     # Set the first n elements of the i-th sample in the mask tensor to 1
                #     mask[i, :n, :] = 1

                x_hat, mu, logvar = cvae(x, label)

                # Create a boolean mask where padding positions are False and non-padding positions are True
                # bool_mask = (mask == 1)

                # Apply the boolean mask to x_hat and x
                # x_hat_masked = x_hat[bool_mask]
                # x_masked = x[bool_mask]

                # loss = loss_fn(x_hat_masked, x_masked)
                loss = loss_fn(x_hat, x)

                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss += kl_loss
                val_loss += loss.item()

            val_loss /= len(val_loader)
            
            # 在每個 epoch 結束時，使用驗證損失來調整學習率
            scheduler.step(val_loss)

            # 打印出當前的學習率
            current_lr = scheduler._last_lr[0]
            print(f"Epoch {epoch+1}/{max_epochs}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, lr: {current_lr:.6f}")
            
            # Call early stopping
            early_stopping(val_loss, cvae)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    torch.save(cvae.state_dict(), save_path)

def generate(num_samples, label_path, save_path):
    cvae = CVAE(input_size=input_size, label_size = label_size, output_size=input_size, num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads, dropout=dropout, latent_size=latent_size, max_len = 40, device=device)
    cvae.load_state_dict(torch.load(save_path))
    cvae = cvae.to(device)
    cvae.eval()
    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, latent_size).to(device)
            label_file_path =  "train/r_dribble_7_17.csv"
            if os.path.exists(label_file_path):
                label = pd.read_csv(label_file_path).values
                frame_len = int(label[0][6])

                label = torch.tensor(label).float().to(device)
                x_hat = cvae.decode(z, label)
                # save pkl
                save_path = f"train/generated_sample_{i+1}.pkl"
                # with open(save_path, 'wb') as f:
                #     pickle.dump(x_hat.squeeze().tolist(), f)
                data = np.array(x_hat.squeeze().tolist()).reshape(-1, 15, 3)
                # 印出date到frame_len的資料
                print(f"frame_len: {frame_len}")
                print(f"data.shape: {data.shape}")
                print(data[:, 0, 0])

                # 取出去掉附檔名的檔案名稱
                file_name, file_extension = os.path.splitext(save_path)

                # 製作完整的 GIF 檔案名稱
                gif_file_name = f"{file_name}.gif"
                # print("save as =>", gif_file_name)
                # 存儲 GIF 檔案
                visualize_and_save(data, frame_len, 'r_dribble_7_17_new.gif')
                
            else:
                print(f"Label file for sample {i+1} not found.")


data_path = "data/split_train_data/pkl"
label_path = "data/split_train_data/csv"
save_path = "train/cvae.pth"
max_epochs = 300
batch_size = 32
num_layers = 2
hidden_size = 512
num_heads = 5
dropout = 0.1
latent_size = 128

input_size = 45
label_size = 7

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 創建一個 SummaryWriter 對象
# writer = SummaryWriter('runs/experiment_1')

# 假設你的模型是 cvae，你需要一個輸入樣本 x 和標籤 label
# x = torch.randn(1, 40, input_size).to(device)
# label = torch.randn(1, 1, label_size).to(device)

# cvae = CVAE(input_size=input_size, label_size = label_size, output_size=input_size, num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads, dropout=dropout, latent_size=latent_size, max_len = 40, device=device)
# cvae = cvae.to(device)  # 將模型移動到正確的設備上
# cvae.load_state_dict(torch.load(save_path))

# 寫入模型
# writer.add_graph(cvae, [x, label])
# writer.close()



# 訓練模型
# train(data_path, label_path, max_epochs, batch_size, num_layers, hidden_size, num_heads, dropout, latent_size, save_path=save_path)

# 生成新序列
generate(num_samples=1, label_path=label_path, save_path=save_path)
