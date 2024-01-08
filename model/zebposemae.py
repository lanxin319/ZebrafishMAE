import torch
import torch.nn as nn
# from .sta_block import STA_Block
# from .pos_embed import Pos_Embed
from model.sta_block import STA_Block
from model.pos_embed import Pos_Embed
import numpy as np
import matplotlib.pyplot as plt


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


class ZebPoseMAE(nn.Module):
    def __init__(self, len_parts, num_frames, num_joints,
                 num_heads, kernel_size, num_fish,
                 num_channels, encoder_config, decoder_config,
                 use_pes=True, att_drop=0):
        super().__init__()

        # Mask ratios
        self.frame_mask_ratio = 0.4
        self.joint_mask_ratio = 0.5
        self.len_parts = len_parts  # len_parts就是consecutive frames的数量 =4
        in_channels = encoder_config[0][0]  # input channels = 64 (第一层)
        decoder_embed_dim = decoder_config[0][0]  # 256
        self.out_channels = decoder_config[-1][1]  # output channels = 2

        self.pos_embed = Pos_Embed(2, num_frames, num_joints)
        self.decoder_pos_embed = Pos_Embed(decoder_embed_dim, num_frames, num_joints)  # fixed sin-cos embedding

        num_frames = num_frames // len_parts  # 总帧数必须要可以整除len_parts
        num_joints = num_joints * len_parts  # 13*6 78

        # 位置编码
        self.pes = Pos_Embed(in_channels, num_frames, num_joints)

        self.input_map = nn.Sequential(
            nn.Conv2d(num_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1))

        # 全为0的mask token，跟decoder的维度相等
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))

        # 构建编码器部分
        self.encoder = nn.ModuleList()
        for in_channels, out_channels, qkv_dim in encoder_config:
            self.encoder.append(
                STA_Block(in_channels, out_channels, qkv_dim,
                          num_frames=int(np.ceil(self.frame_mask_ratio*(num_frames/len_parts))*len_parts),
                          num_joints=int(np.ceil(self.joint_mask_ratio*(num_joints/len_parts))*len_parts),
                          num_heads=num_heads,
                          kernel_size=kernel_size,
                          use_pes=use_pes,
                          att_drop=att_drop,
                          is_encoder=True)
            )

        # 构建解码器部分
        self.decoder = nn.ModuleList()
        for in_channels, out_channels, qkv_dim in decoder_config:
            self.decoder.append(
                STA_Block(in_channels, out_channels, qkv_dim,
                          num_frames=num_frames,
                          num_joints=num_joints,
                          num_heads=num_heads,
                          kernel_size=kernel_size,
                          use_pes=use_pes,
                          att_drop=att_drop,
                          is_encoder=False)
            )

        # self.modules() 是 torch.nn.Module 类的一个内置方法。
        # 可以调用 self.modules() 来递归地遍历模型中所有的模块和子模块。这个方法返回一个迭代器，包含模型中的所有模块，包括模型本身。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, scale=1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def mask_frames(self, x, mask_ratio):
        """
        对输入张量 x 进行每个样本的随机遮蔽。
        x: [N, C, T, V], 其中 N 是批次大小，C 是通道数，T 是帧数，V 是每帧的特征维度。
        """
        N, C, T, V = x.shape  # batch, channel, frame, joints
        len_keep = int(T * (1 - mask_ratio))  # len_keep = 72

        # 为每个样本生成随机噪声，[32, 120]
        noise = torch.rand(N, T, device=x.device)  # 噪声在 [0, 1]
        # 为每个样本排序噪声，[32, 120]，存的都是索引
        # 通过对每个样本的随机噪声进行排序得到的，用于决定哪些帧被保留。这个过程会根据噪声的值改变帧的顺序。
        ids_shuffle = torch.argsort(noise, dim=1)  # 升序：小的保留，大的移除

        # 保留ids_keep部分子集
        ids_keep = torch.sort(ids_shuffle[:, :len_keep], dim=1)[0]  # 【32，72】
        ids_mask = torch.sort(ids_shuffle[:, len_keep:], dim=1)[0]
        ids_combined = torch.cat([ids_keep, ids_mask], dim = 1)
        # x_masked.shape [32, 2, 72, 13]
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(1).unsqueeze(-1).expand(-1, C, -1, V))

        return x_masked, ids_combined

    def reconstruct_data(self, encoded_data, frame_ids_restore, joint_ids_restore):
        Ne, Ce, Te, Ve = encoded_data.shape  # encoded_data的维度, (32, 256, 12, 42)

        # 先把数据从[32, 256, 12, 42]变回 [32, 256, 72, 7]
        encoded_data = encoded_data.view(encoded_data.size(0), encoded_data.size(1), Te*self.len_parts, Ve//self.len_parts)
        # 然后把(N, C, T, V)变成(N, T, V, C)
        encoded_data = encoded_data.permute(0, 2, 3, 1)

        # 还原关节:
        # 创建一个形状为 [32, 72, 6, 256] 的零张量
        zeros_to_add = torch.zeros(Ne, encoded_data.size(1), 6, Ce, device=encoded_data.device)
        # 把零张量加在encoded_data后面，把形状还原成[32, 72, 13, 256]
        encoded_data = torch.cat((encoded_data, zeros_to_add), dim=2)

        # 创建一个用于批量索引的辅助张量
        batch_indices = torch.arange(Ne)[:, None, None]  # 形状 [N, 1, 1]
        time_indices = torch.arange(encoded_data.size(1))[None, :, None]  # 形状 [1, T, 1]

        # 使用高级索引还原关节顺序
        reconstructed_data = encoded_data[batch_indices, time_indices, joint_ids_restore]

        # 还原帧:
        zeros_to_add = torch.zeros(Ne, 48, 13, Ce, device=encoded_data.device)
        # 把形状还原成[32, 120, 13, 256]
        reconstructed_data = torch.cat((reconstructed_data, zeros_to_add), dim=1)

        batch_indices = torch.arange(Ne)[:, None]  # 形状 [N, 1]
        # (32, 120, 13, 256)
        reconstructed_data = reconstructed_data[batch_indices, frame_ids_restore]
        reconstructed_data = reconstructed_data.permute(0, 3, 1, 2)

        return reconstructed_data

    def mask_joints(self, x, mask_ratio):
        """
        针对给定的已经被mask frame的x，随机选择一定比例的关节，并将这些关节的所有元素拿走。
        每一帧上被mask的关节都是随机的
        """
        N, C, T, V = x.shape  # batch, channel, frame, joints
        num_joints_to_mask = int(V * mask_ratio)  # 每一帧要遮蔽的关节数量:6

        # 初始化joint_mask张量，刚开始全部都是False，[32, 72, 13]
        joint_mask = torch.zeros(N, T, V, dtype=torch.bool, device=x.device)
        ids_restore = torch.zeros_like(joint_mask, dtype=torch.long, device=x.device)
        for n in range(N):
            for t in range(T):
                # 首先为每一帧 (T) 定义了一个遮蔽模式，这个遮蔽模式在每一帧中都是独立进行的
                mask_indices = torch.randperm(V, device=x.device)[:num_joints_to_mask]
                joint_mask[n, t, mask_indices] = True

        x = x.permute(0, 2, 3, 1)  # 变成了N, T, V, C
        # 初始化 x_masked 张量
        x_masked = torch.zeros(N, T, V - num_joints_to_mask, C, device=x.device)  # (32, 72, 7, 2)
        for n in range(N):
            for t in range(T):
                # 获取当前帧的保留关节索引
                ids_keep_nt = torch.where(~joint_mask[n, t])[0]
                ids_mask_nt = torch.where(joint_mask[n, t])[0]
                combined_nt = torch.cat([ids_keep_nt, ids_mask_nt])
                ids_restore[n, t, :combined_nt.size(0)] = combined_nt
                # 选择保留的关节
                x_masked[n, t] = x[n, t, ids_keep_nt]

        x_masked = x_masked.permute(0, 3, 1, 2)  # （32，2，72，7）

        return x_masked, joint_mask, ids_restore

    def forward(self, x):

        N, C, T, V, M = x.shape  # Shape:[32, 2, 120, 13, 1]
        # x = x.to('mps')

        # torch.Size([32, 2, 120, 13])
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        x_copy = x.clone()
        x = x + self.pos_embed(x)  # 做一下pe

        # 在sequence division之前做masking:
        # frame masking，[32, 2, 72, 13]
        x_masked, frame_ids_restore = self.mask_frames(x, self.frame_mask_ratio)
        # joint masking，[32, 2, 72, 7]
        x_masked, joint_mask, joint_ids_restore = self.mask_joints(x_masked, self.joint_mask_ratio)

        Nm, Cm, Tm, Vm = x_masked.shape  # New Shape:[32, 2, 72, 7]

        # Sequence division
        # 这里的Tm和Vm应该是mask之后的大小, (32,2,12,42)
        x = x_masked.view(x_masked.size(0), x_masked.size(1), Tm // self.len_parts, Vm * self.len_parts)
        # x = x.view(x.size(0), x.size(1), T // self.len_parts, V * self.len_parts)
        x = self.input_map(x)  # (32, 64, 12, 42)

        masked_data = x  # New Shape: [32, 2, 72/6, 7*6] = [32, 2, 12, 42]

        # Encoder
        for block in self.encoder:
            masked_data = block(masked_data)

        # masked_data从encoder里出来以后维度应该是:[32, 256, 12, 42]
        # Reconstruction 要把数据重构回[32, 256, 20, 78]
        reconstructed_data = self.reconstruct_data(masked_data, frame_ids_restore, joint_ids_restore)
        # 加上原始数据x的pe？
        decoded_data = reconstructed_data + self.decoder_pos_embed(x_copy)
        decoded_data = decoded_data.view(N, decoded_data.size(1), T // self.len_parts, V * self.len_parts)

        # Decoder
        for block in self.decoder:
            decoded_data = block(decoded_data)

        decoded_data = decoded_data.unsqueeze(-1)
        decoded_data = decoded_data.view(N, C, T, V, M)

        return decoded_data

len_parts = 6
num_frames = 120
num_joints = 13
num_heads = 3
kernel_size = [3, 5]
num_fish = 1
num_channels = 2
encoder_config = [[64, 64, 16], [64, 64, 16], [64, 128, 32], [128, 128, 32],
                 [128, 256, 64], [256, 256, 64], [256, 256, 64], [256, 256, 64]]
decoder_config = [[256, 256, 64], [256, 256, 64], [256, 256, 64], [256, 256, 64],
                 [256, 128, 32], [128, 128, 32], [128, 64, 16], [64, 64, 16],
                 [64, 2, 16]]

data = torch.randn(32, 2, 120, 13, 1)
model = ZebPoseMAE(len_parts, num_frames, num_joints,
                 num_heads, kernel_size, num_fish,
                 num_channels, encoder_config, decoder_config)

model.forward(data)

print(data)



