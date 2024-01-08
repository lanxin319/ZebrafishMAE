import torch
import torch.nn as nn
import math
import numpy as np


class Pos_Embed(nn.Module):
    # channels表示位置编码的维度，num_frames表示序列中的帧数，num_joints表示每帧中的关节数量
    def __init__(self, channels, num_frames, num_joints):
        super().__init__()

        # 首先创建一个pos_list列表，其中包含了序列中每个关节的位置信息。这个列表的长度将是num_frames * num_joints
        pos_list = []
        for tk in range(num_frames):
            for st in range(num_joints):
                pos_list.append(st)

        # 使用torch.from_numpy()将NumPy数组转换为PyTorch张量
        # 并使用.unsqueeze(1)将其形状从(num_frames * num_joints,)变为(num_frames * num_joints, 1)
        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        pe = torch.zeros(num_frames * num_joints, channels)

        # 一种常见的位置编码方法，即将正弦和余弦函数的值嵌入到位置编码中，以捕捉序列中的位置信息。
        # 创建一个名为div_term的张量，其中包含了一系列递增的值，这些值用于缩放正弦和余弦函数的周期。
        div_term = torch.exp(torch.arange(0, channels, 2).float() * -(math.log(10000.0) / channels))

        # 使用pe[:, 0::2]和pe[:, 1::2]分别填充pe张量的偶数和奇数列，以存储正弦和余弦函数的值，这样可以捕捉不同位置的信息。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 将pe张量的形状从(num_frames * num_joints, channels)转换为(channels, num_frames, num_joints)
        # 并在前面添加一个维度，以得到形状为(1, channels, num_frames, num_joints)的张量。这个张量表示了位置编码。
        pe = pe.view(num_frames, num_joints, channels).permute(2, 0, 1).unsqueeze(0)
        # 将位置编码张量注册为模块的缓冲区（buffer），以便在模型的前向传播中使用。
        self.register_buffer('pe', pe)

    # 模块的前向传播函数。它接受输入张量x，通常是模型的输入数据。
    def forward(self, x):  # nctv
        # 在前向传播过程中，它将位置编码张量pe的形状调整为与输入张量x相匹配，以便将位置编码与输入数据相加。
        x = self.pe[:, :, :x.size(2)]
        # 这样，位置信息就被嵌入到输入数据中，并在模型中传递。
        return x