import torch
import torch.nn as nn

class GaitTransformer(nn.Module):
    """
    输入: batch_size x seq_len x n_joints x 3
    输出: batch_size x seq_len x n_features (可解释步态参数)
    """
    def __init__(self, n_joints=8, in_dim=3, d_model=128, nhead=4, num_layers=3, dropout=0.1, out_dim=10):
        super().__init__()
        self.n_joints = n_joints
        self.in_dim = in_dim
        self.input_linear = nn.Linear(n_joints*in_dim, d_model)  # 每帧展平成向量
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, out_dim)  # 输出步态参数

    def forward(self, x):
        B, T, J, C = x.shape
        x = x.view(B, T, J*C)
        x = self.input_linear(x)
        x = self.transformer(x)
        out = self.output_linear(x)
        return out