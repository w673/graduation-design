import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from gait_transformer_model import GaitTransformer
from pose_estimator_3d import PoseEstimator3D, PoseEstimator  # 你自己的3D骨架类

class GaitDataset(Dataset):
    def __init__(self, images_list, depth_list, targets=None, pose3d=None):
        """
        images_list: list of N_frames x H x W x 3 numpy array
        depth_list:  list of N_frames x H x W numpy array
        targets: numpy array N_samples x seq_len x n_features
        pose3d: PoseEstimator3D实例
        """
        self.pose3d = pose3d
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None

        # 预处理视频生成3D骨架
        skeletons_all = []
        for imgs, depths in zip(images_list, depth_list):
            skeletons = self.pose3d.detect_batch_3d(imgs, depths)  # seq_len x n_joints x 3
            skeletons_all.append(skeletons)
        self.skeletons = torch.tensor(np.array(skeletons_all), dtype=torch.float32)

    def __len__(self):
        return len(self.skeletons)

    def __getitem__(self, idx):
        x = self.skeletons[idx]
        if self.targets is not None:
            y = self.targets[idx]
            return x, y
        else:
            return x

def train_model(model, train_loader, val_loader=None, lr=1e-4, epochs=50, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        total_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss:.4f}")

        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_val, Y_val in val_loader:
                    X_val, Y_val = X_val.to(device), Y_val.to(device)
                    out_val = model(X_val)
                    loss_val = criterion(out_val, Y_val)
                    val_loss += loss_val.item() * X_val.size(0)
            val_loss /= len(val_loader.dataset)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'best_gait_transformer.pth')
                print("Saved best model.")

if __name__ == "__main__":
    # 示例: 随机生成RGB和深度数据
    images_list = [np.random.rand(50, 64, 64, 3) for _ in range(5)]
    depth_list  = [np.random.rand(50, 64, 64) for _ in range(5)]
    targets     = np.random.rand(5, 50, 10)

    pose2d = PoseEstimator()  # 你自己定义的2D pose estimator
    pose3d = PoseEstimator3D(pose2d, depth_scale=1.0)
    pose3d.set_camera_intrinsics(500,500,32,32)

    train_dataset = GaitDataset(images_list, depth_list, targets, pose3d)
    train_loader  = DataLoader(train_dataset, batch_size=2, shuffle=True)

    model = GaitTransformer(n_joints=8, in_dim=3, d_model=128, nhead=4, num_layers=3, out_dim=10)
    train_model(model, train_loader, val_loader=None, lr=1e-4, epochs=3)