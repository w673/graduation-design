import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=2000):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


class GaitTransformer(nn.Module):
    """
    输入:
        B x T x J x 3

    输出:
        B x T x out_dim

    默认:
        out_dim = 13
    """

    def __init__(
        self,
        n_joints=9,
        in_dim=3,
        d_model=128,
        nhead=4,
        num_layers=3,
        dropout=0.1,
        out_dim=13
    ):

        super().__init__()

        self.n_joints = n_joints
        self.in_dim = in_dim

        input_size = n_joints * in_dim

        # skeleton embedding
        self.embedding = nn.Sequential(

            nn.Linear(input_size, d_model),
            nn.ReLU(),

            nn.Linear(d_model, d_model)
        )

        # positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # output head
        self.head = nn.Sequential(

            nn.Linear(d_model, 128),
            nn.ReLU(),

            nn.Linear(128, out_dim)
        )

    def forward(self, x):

        """
        x: B x T x J x 3
        """

        B, T, J, C = x.shape

        # flatten skeleton
        x = x.view(B, T, J * C)

        # embedding
        x = self.embedding(x)

        # positional encoding
        x = self.pos_encoding(x)

        # transformer
        x = self.transformer(x)

        # prediction
        out = self.head(x)

        return out