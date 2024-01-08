# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os.path as osp
import os
import numpy as np
import pickle
import logging


def get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger):
    """
    Get raw bodies data from a skeleton sequence.

    Each body's data is a dict that contains the following keys:
      - joints: raw 2D joints positions. Shape: (num_frames x 13, 2)
      - interval: a list which stores the frame indices of this body.

    Return:
      a dict for a skeleton sequence with 3 key-value pairs:
        - name: the skeleton filename.
        - data: a dict which stores raw data of each body.
        - num_frames: the number of valid frames.
    """

    ske_file = osp.join(skes_path, ske_name + '.skeleton')
    assert osp.exists(ske_file), 'Error: Skeleton file %s not found' % ske_file
    # Read all data from .skeleton file into a list (in string format)
    print('Reading data from %s' % ske_file)
    with open(ske_file, 'r') as fr:
        str_data = fr.readlines()

    num_frames = int(str_data[0].strip('\r\n'))
    frames_drop = []  # 用于记录丢帧信息
    body_data = {'joints': [], 'interval': []}
    valid_frames = 0  # 0-based index for valid frames
    current_line = 1

    for f in range(num_frames):
        num_bodies = int(str_data[current_line].strip('\r\n'))
        current_line += 2

        if num_bodies == 0:  # no data in this frame, drop it
            frames_drop.append(f)  # 0-based index
            continue

        # 初始化当前帧的关节数组
        joints = np.zeros((13, 2), dtype=np.float32)  # 修改为(13, 2)因为每个关节只有x,y坐标
        num_joints = 13

        for j in range(num_joints):
            temp_str = str_data[current_line].strip('\r\n').split()
            joints[j, :] = np.array(temp_str[:2], dtype=np.float32)  # 取前两个数值
            current_line += 1

        body_data['joints'].append(joints)
        body_data['interval'].append(f)  # 使用f作为帧索引

        valid_frames += 1  # 递增有效帧计数

    num_frames_drop = len(frames_drop)
    assert num_frames_drop < num_frames, \
        'Error: All frames data (%d) of %s is missing or lost' % (num_frames, ske_name)
    if num_frames_drop > 0:
        frames_drop_skes[ske_name] = np.array(frames_drop, dtype=int)
        frames_drop_logger.info('{}: {} frames missed: {}\n'.format(ske_name, num_frames_drop,
                                                                    frames_drop))

    # 最后将joints列表转换为一个3维数组
    body_data['joints'] = np.array(body_data['joints'], dtype=np.float32)  # 将会是(num_valid_frames, 13, 2)的形状

    return {'name': ske_name, 'data': body_data, 'num_frames': valid_frames}


def get_raw_skes_data():

    skes_name = np.loadtxt(skes_name_file, dtype=str)

    num_files = skes_name.size
    print('Found %d available skeleton files.' % num_files)

    raw_skes_data = []
    frames_cnt = np.zeros(num_files, dtype=int)

    for (idx, ske_name) in enumerate(skes_name):
        bodies_data = get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger)
        raw_skes_data.append(bodies_data)
        frames_cnt[idx] = bodies_data['num_frames']
        if (idx + 1) % 1000 == 0:
            print('Processed: %.2f%% (%d / %d)' % \
                  (100.0 * (idx + 1) / num_files, idx + 1, num_files))

    with open(save_data_pkl, 'wb') as fw:
        pickle.dump(raw_skes_data, fw, pickle.HIGHEST_PROTOCOL)
    np.savetxt(osp.join(save_path, 'raw_data', 'frames_cnt.txt'), frames_cnt, fmt='%d')

    print('Saved raw bodies data into %s' % save_data_pkl)
    print('Total frames: %d' % np.sum(frames_cnt))

    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    save_path = './'

    skes_path = '../zebrafish_raw/skeleton_files/'
    stat_path = osp.join(save_path, 'statistics')
    # 创建存储raw data的文件夹
    if not osp.exists('./raw_data'):
        os.makedirs('./raw_data')

    # 这个路径是存储所有文件的名字的txt，我自己也可以生成一个
    skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
    # 用于保存原始骨架数据
    save_data_pkl = osp.join(save_path, 'raw_data', 'raw_skes_data.pkl')
    # 用于记录丢帧的信息
    frames_drop_pkl = osp.join(save_path, 'raw_data', 'frames_drop_skes.pkl')

    frames_drop_logger = logging.getLogger('frames_drop')
    frames_drop_logger.setLevel(logging.INFO)
    frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'raw_data', 'frames_drop.log')))
    # 初始化一个字典，用于存储关于丢帧的信息
    frames_drop_skes = dict()

    get_raw_skes_data()

    # 打开pickle文件，并使用pickle.dump将frames_drop_skes字典保存到文件中。
    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)