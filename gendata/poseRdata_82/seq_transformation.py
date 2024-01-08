# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import os.path as osp
import numpy as np
import pickle

root_path = './'
stat_path = osp.join(root_path, 'statistics')
skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
denoised_path = osp.join(root_path, 'denoised_data')
raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
frames_file = osp.join(denoised_path, 'frames_cnt.txt')

save_path = './'


if not osp.exists(save_path):
    os.mkdir(save_path)


def seq_translation(skes_joints):
    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_joints = 13  # 我的数据中有13个关节

        i = 0  # 寻找第一个非零帧
        while i < num_frames:
            if np.any(ske_joints[i] != 0):
                break
            i += 1

        origin = np.copy(ske_joints[i, 2:4])  # 新原点: 第二个关节，可以改

        for f in range(num_frames):
            ske_joints[f] -= np.tile(origin, num_joints)  # 平移到新原点

        skes_joints[idx] = ske_joints  # 更新

    return skes_joints


def split_dataset(skes_joints, split_ratio=0.8, save_path='.'):
    skes_joints = np.array(skes_joints)

    num_samples = len(skes_joints)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # 计算训练集和测试集的大小
    train_size = int(num_samples * split_ratio)
    test_size = num_samples - train_size

    # 分割数据集
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_x = skes_joints[train_indices]
    test_x = skes_joints[test_indices]

    save_name = osp.join(save_path, 'Zebrafish210.npz')
    np.savez(save_name, x_train=train_x, x_test=test_x)

    print(f"数据集已保存到 {save_name}")
    print(f"训练集大小: {train_size}, 测试集大小: {test_size}")


if __name__ == '__main__':
    # 从文件中加载每个骨架序列的帧数
    frames_cnt = np.loadtxt(frames_file, dtype=int)  # frames count
    # 加载骨架序列文件的名称。
    skes_name = np.loadtxt(skes_name_file, dtype=np.string_)

    # 使用 pickle 加载去噪后的骨架关节数据（skes_joints），这是一个列表。
    # skes_joints的长度等于bout数量，里面的每一个元素都是(210, 26)的数组
    # (210, 26)的数组代表210帧，和里面被展开了的关节坐标
    with open(raw_skes_joints_pkl, 'rb') as fr:
        skes_joints = pickle.load(fr)  # a list

    # 序列转换，目的是将每个骨架序列的原点平移到第二个关节的位置。
    # 用于归一化骨架数据，使其不受全局位置变化的影响。
    skes_joints = seq_translation(skes_joints)

    # 划分数据集
    split_dataset(skes_joints)
