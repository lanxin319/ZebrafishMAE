import numpy as np
import torch
import torch.nn as nn
from .pos_embed import Pos_Embed


class STA_Block(nn.Module):
    # 使用了自注意力机制（self-attention mechanism）来捕获关节之间以及不同帧之间的关系。
    def __init__(self, in_channels, out_channels, qkv_dim,
                 num_frames, num_joints, num_heads,
                 kernel_size, use_pes=True, att_drop=0, is_encoder=False):
        super().__init__()
        self.qkv_dim = qkv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_pes = use_pes

        self.is_encoder = is_encoder  # 新加的

        pads = int((kernel_size[1] - 1) / 2)
        padt = int((kernel_size[0] - 1) / 2)

        # Spatio-Temporal Tuples Attention
        if self.use_pes:
            self.pes = Pos_Embed(in_channels, num_frames, num_joints)

            # # 新加的：
            # if self.is_encoder:
            #     # 在encoder中调整关节的数量
            #     adjusted_num_joints = int(num_joints * 0.5)
            #     self.pes = Pos_Embed(in_channels, num_frames, adjusted_num_joints)
            # else:
            #     self.pes = Pos_Embed(in_channels, num_frames, num_joints)

        # 利用一个卷积层 to_qkvs 生成查询（query）和键（key）向量。
        self.to_qkvs = nn.Conv2d(in_channels, 2 * num_heads * qkv_dim, 1, bias=True)
        self.alphas = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)

        # # 新加的：
        # if self.is_encoder:
        #     self.att0s = nn.Parameter(torch.ones(1, num_heads, int(num_joints * 0.5), int(num_joints * 0.5)) / num_joints, requires_grad=True)
        # else:
        #     self.att0s = nn.Parameter(torch.ones(1, num_heads, num_joints, num_joints) / num_joints, requires_grad=True)

        self.att0s = nn.Parameter(torch.ones(1, num_heads, num_joints, num_joints) / num_joints, requires_grad=True)

        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * num_heads, out_channels, (1, kernel_size[1]), padding=(0, pads)),
            nn.BatchNorm2d(out_channels))
        self.ff_net = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))

        # Inter-Frame Feature Aggregation
        self.out_nett = nn.Sequential(nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), padding=(padt, 0)),
                                      nn.BatchNorm2d(out_channels))

        if in_channels != out_channels:
            self.ress = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            self.rest = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.ress = lambda x: x
            self.rest = lambda x: x

        # 计算注意力权重，使用 tanh 函数来归一化。
        # 应用注意力权重到输入数据，以获得特征的加权组合。
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(att_drop)

    def forward(self, x):

        N, C, T, V = x.size()
        # Spatio-Temporal Tuples Attention
        xs = self.pes(x) + x if self.use_pes else x

        # if self.use_pes:
        #     pes_output = self.pes(x).to(x.device)
        #     xs = pes_output + x
        # else:
        #     xs = x

        q, k = torch.chunk(self.to_qkvs(xs).view(N, 2 * self.num_heads, self.qkv_dim, T, V), 2, dim=1)
        attention = self.tan(torch.einsum('nhctu,nhctv->nhuv', [q, k]) / (self.qkv_dim * T)) * self.alphas
        attention = attention + self.att0s.repeat(N, 1, 1, 1)
        attention = self.drop(attention)
        xs = torch.einsum('nctu,nhuv->nhctv', [x, attention]).contiguous().view(N, self.num_heads * self.in_channels, T,
                                                                                V)
        x_ress = self.ress(x)
        xs = self.relu(self.out_nets(xs) + x_ress)
        xs = self.relu(self.ff_net(xs) + x_ress)

        # Inter-Frame Feature Aggregation
        # 使用卷积层 out_nett 来聚合不同帧之间的特征。
        xt = self.relu(self.out_nett(xs) + self.rest(xs))

        return xt