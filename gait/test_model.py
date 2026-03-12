import torch
from torch.utils.data import DataLoader
from gait_transformer_model import GaitTransformer
from train_model import GaitDataset
from pose_estimator_3d import PoseEstimator3D, PoseEstimator
import numpy as np

def predict(model, images_list, depth_list, pose3d, device='cuda'):
    model.eval()
    dataset = GaitDataset(images_list, depth_list, targets=None, pose3d=pose3d)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    preds = []
    with torch.no_grad():
        for X in loader:
            X = X.to(device)
            out = model(X)
            preds.append(out.cpu())
    return torch.cat(preds, dim=0).numpy()

if __name__ == "__main__":
    images_list = [np.random.rand(50,64,64,3)]
    depth_list  = [np.random.rand(50,64,64)]

    pose2d = PoseEstimator()
    pose3d = PoseEstimator3D(pose2d, depth_scale=1.0)
    pose3d.set_camera_intrinsics(500,500,32,32)

    model = GaitTransformer(n_joints=8, in_dim=3, d_model=128, nhead=4, num_layers=3, out_dim=10)
    model.load_state_dict(torch.load('best_gait_transformer.pth', map_location='cpu'))

    predicted_params = predict(model, images_list, depth_list, pose3d)
    print("Predicted params shape:", predicted_params.shape)