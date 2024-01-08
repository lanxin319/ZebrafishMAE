# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import os.path as osp
import numpy as np
import pickle
import logging

root_path = './'
raw_data_file = osp.join(root_path, 'raw_data', 'raw_skes_data.pkl')
save_path = osp.join(root_path, 'denoised_data')

if not osp.exists(save_path):
    os.mkdir(save_path)

rgb_ske_path = osp.join(save_path, 'rgb+ske')
if not osp.exists(rgb_ske_path):
    os.mkdir(rgb_ske_path)

actors_info_dir = osp.join(save_path, 'actors_info')
if not osp.exists(actors_info_dir):
    os.mkdir(actors_info_dir)

missing_count = 0
noise_len_thres = 11
noise_spr_thres1 = 0.8
noise_spr_thres2 = 0.69754
noise_mot_thres_lo = 0.089925
noise_mot_thres_hi = 2

noise_len_logger = logging.getLogger('noise_length')
noise_len_logger.setLevel(logging.INFO)
noise_len_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_length.log')))
noise_len_logger.info('{:^20}\t{:^17}\t{:^8}\t{}'.format('Skeleton', 'bodyID', 'Motion', 'Length'))

noise_spr_logger = logging.getLogger('noise_spread')
noise_spr_logger.setLevel(logging.INFO)
noise_spr_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_spread.log')))
noise_spr_logger.info('{:^20}\t{:^17}\t{:^8}\t{:^8}'.format('Skeleton', 'bodyID', 'Motion', 'Rate'))

noise_mot_logger = logging.getLogger('noise_motion')
noise_mot_logger.setLevel(logging.INFO)
noise_mot_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_motion.log')))
noise_mot_logger.info('{:^20}\t{:^17}\t{:^8}'.format('Skeleton', 'bodyID', 'Motion'))

fail_logger_1 = logging.getLogger('noise_outliers_1')
fail_logger_1.setLevel(logging.INFO)
fail_logger_1.addHandler(logging.FileHandler(osp.join(save_path, 'denoised_failed_1.log')))

fail_logger_2 = logging.getLogger('noise_outliers_2')
fail_logger_2.setLevel(logging.INFO)
fail_logger_2.addHandler(logging.FileHandler(osp.join(save_path, 'denoised_failed_2.log')))

missing_skes_logger = logging.getLogger('missing_frames')
missing_skes_logger.setLevel(logging.INFO)
missing_skes_logger.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes.log')))
missing_skes_logger.info('{:^20}\t{}\t{}'.format('Skeleton', 'num_frames', 'num_missing'))

missing_skes_logger1 = logging.getLogger('missing_frames_1')
missing_skes_logger1.setLevel(logging.INFO)
missing_skes_logger1.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes_1.log')))
missing_skes_logger1.info('{:^20}\t{}\t{}\t{}\t{}\t{}'.format('Skeleton', 'num_frames', 'Actor1',
                                                              'Actor2', 'Start', 'End'))

missing_skes_logger2 = logging.getLogger('missing_frames_2')
missing_skes_logger2.setLevel(logging.INFO)
missing_skes_logger2.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes_2.log')))
missing_skes_logger2.info('{:^20}\t{}\t{}\t{}'.format('Skeleton', 'num_frames', 'Actor1', 'Actor2'))


def denoising_by_length(ske_name, bodies_data):
    """
    Denoising data based on the frame length for each bodyID.
    Filter out the bodyID which length is less or equal than the predefined threshold.

    """
    noise_info = str()
    new_bodies_data = bodies_data.copy()
    for (bodyID, body_data) in new_bodies_data.items():
        length = len(body_data['interval'])
        if length <= noise_len_thres:
            noise_info += 'Filter out: %s, %d (length).\n' % (bodyID, length)
            noise_len_logger.info('{}\t{}\t{:.6f}\t{:^6d}'.format(ske_name, bodyID,
                                                                  body_data['motion'], length))
            del bodies_data[bodyID]
    if noise_info != '':
        noise_info += '\n'

    return bodies_data, noise_info


def get_valid_frames_by_spread(points):
    """
    Find the valid (or reasonable) frames (index) based on the spread of X and Y.

    points: joints or colors
    """
    num_frames = points.shape[0]
    valid_frames = []
    for i in range(num_frames):
        x = points[i, :, 0]
        y = points[i, :, 1]
        if (x.max() - x.min()) <= noise_spr_thres1 * (y.max() - y.min()):  # 0.8
            valid_frames.append(i)
    return valid_frames


def get_one_actor_points(body_data, num_frames):
    """
    Get joints for only one actor.
    Each frame contains 26 X-Y coordinates (13 joints with X and Y coordinates).
    """
    # Initialize the joints array for 13 joints and 2 coordinates (X, Y)
    joints = np.zeros((num_frames, 26), dtype=np.float32)
    start, end = body_data['interval'][0], body_data['interval'][-1]

    # Reshape the joints data and fill it into the initialized array
    # Assuming 'body_data['joints']' is a 2D array with shape (num_frames, 13, 2)
    joints[start:end + 1] = body_data['joints'].reshape(-1, 26)

    # Since there is no color data in the provided structure, return only the joints
    return joints


def remove_missing_frames(ske_name, joints):
    """
    去除所有关节位置都为0的帧，这通常表示数据丢失或损坏。

    参数:
    - ske_name: 骨架序列的名称。
    - joints: 关节位置数据，形状为 (num_frames, 26)。

    返回:
    - 更新后的关节位置数据。
    """
    num_frames = joints.shape[0]

    # 找到有效帧的索引，即数据不丢失的帧
    valid_indices = np.where(joints.sum(axis=1) != 0)[0]
    missing_indices = np.where(joints.sum(axis=1) == 0)[0]
    num_missing = len(missing_indices)

    if num_missing > 0:  # 如果有丢失的帧，更新joints数据
        joints = joints[valid_indices]
        global missing_count
        missing_count += 1
        missing_skes_logger.info('{}\t{:^10d}\t{:^11d}'.format(ske_name, num_frames, num_missing))

    return joints


def get_bodies_info(bodies_data):
    bodies_info = '{:^17}\t{}\t{:^8}\n'.format('bodyID', 'Interval', 'Motion')
    for (bodyID, body_data) in bodies_data.items():
        start, end = body_data['interval'][0], body_data['interval'][-1]
        bodies_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), body_data['motion'])

    return bodies_info + '\n'


def get_raw_denoised_data():
    """
    获取去噪后的数据（关节位置）从原始骨架序列。

    对于骨架序列中的每一帧，一个演员的13个关节的2D位置用一个26维向量表示，
    通过沿着行维度连接每个2维（x, y）坐标。每个26维向量作为行向量放入一个2D numpy数组，
    其中行数等于有效帧的数量。所有这些2D数组被放入一个列表中，最后将列表序列化为cPickle文件。

    缺失的帧也被记录在日志文件中。
    """

    with open(raw_data_file, 'rb') as fr:  # 加载原始骨架数据
        raw_skes_data = pickle.load(fr)

    num_skes = len(raw_skes_data)
    print('Found %d available skeleton sequences.' % num_skes)

    raw_denoised_joints = []
    frames_cnt = []

    for (idx, bodies_data) in enumerate(raw_skes_data):
        ske_name = bodies_data['name']
        print('Processing %s' % ske_name)

        num_frames = bodies_data['num_frames']
        # body_data = list(bodies_data['data'].values())[0]
        body_data = bodies_data['data']
        joints = get_one_actor_points(body_data, num_frames)

        # 去除缺失的帧
        joints = remove_missing_frames(ske_name, joints)
        num_frames = joints.shape[0]  # 更新帧数

        raw_denoised_joints.append(joints)
        frames_cnt.append(num_frames)

        if (idx + 1) % 1000 == 0:
            print('Processed: %.2f%% (%d / %d), ' % \
                  (100.0 * (idx + 1) / num_skes, idx + 1, num_skes) + \
                  'Missing count: %d' % missing_count)

    raw_skes_joints_pkl = osp.join(save_path, 'raw_denoised_joints.pkl')
    with open(raw_skes_joints_pkl, 'wb') as f:
        pickle.dump(raw_denoised_joints, f, pickle.HIGHEST_PROTOCOL)

    frames_cnt = np.array(frames_cnt, dtype=int)
    np.savetxt(osp.join(save_path, 'frames_cnt.txt'), frames_cnt, fmt='%d')

    print('Saved raw denoised positions of {} frames into {}'.format(np.sum(frames_cnt),
                                                                     raw_skes_joints_pkl))
    print('Found %d files that have missing data' % missing_count)

if __name__ == '__main__':
    # 原始代码保存下来的每个骨架序列的数据形状为 (num_frames, 150)
    # 即每一帧是一个150维的向量，共有 num_frames 帧。
    # 修改后的代码保存下来的每个骨架序列的数据形状为 (num_frames, 26)
    # 即每一帧是一个26维的向量，共有 num_frames 帧。
    get_raw_denoised_data()
