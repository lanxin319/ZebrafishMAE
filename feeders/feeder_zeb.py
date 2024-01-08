import numpy as np
from torch.utils.data import Dataset
from feeders import tools


# Feeder是一个数据集类，它负责加载和预处理数据以便于之后的训练或测试。
# 目的是按照指定的参数配置加载和预处理数据，使其适合于模型训练和测试。
class Feeder(Dataset):
    def __init__(self, data_path, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=True,
                 bone=False, vel=False):
        """
        data_path:
        split: training set or test set
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move:
        random_rot: rotate skeleton around xyz axis
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
        use_mmap: If true, use mmap mode to load data, which can save the running memory
        bone: use bone modality or not
        vel: use motion modality or not
        only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    # 划分训练集和测试集，并设置数据的维度以适配模型的输入要求。
    def load_data(self):
        # data: N C V T M， (样本数, 坐标维度, 时间序列长度, 关节数, 演员数)
        # N = number of bouts (number of .skeleton files)
        if self.use_mmap:
            # 确定是否使用内存映射（mmap）模式来加载数据。
            # 这种模式意味着数据文件会以内存映射的方式打开，可以让数据在不完全加载到内存中的情况下被访问，有助于处理大型数据集。
            zeb_data = np.load(self.data_path, mmap_mode='r')
        else:
            zeb_data = np.load(self.data_path)

        # 决定加载训练数据还是测试数据。
        if self.split == 'train':
            self.data = zeb_data['x_train']
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = zeb_data['x_test']
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        elif self.split == 'infer':
            X_train = zeb_data['x_train']
            X_test = zeb_data['x_test']
            sample_train = ['train_' + str(i) for i in range(len(X_train))]
            sample_test = ['test_' + str(i) for i in range(len(X_test))]
            self.data = np.concatenate((X_train, X_test), axis=0)
            self.sample_name = sample_train + sample_test  # 列表连接
        else:
            raise NotImplementedError('data split only supports train/test')

        # data原本是3维（可能是(N×T×(V⋅M⋅C) 形式的），重塑以后才变成5维
        N, T, _ = self.data.shape
        # 将数据重塑成一个五维数组
        # N代表bout数，T代表帧数，第四个数字代表关节数，最后一个数字应该代表channel数
        # 这里的2代表了有两个演员，我只有一个，保留了这个维度，但是改成了1
        # self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        self.data = self.data.reshape((N, T, 1, 19, 2)).transpose(0, 4, 1, 3, 2)


    # 如果启用了归一化，这个方法会计算整个数据集的均值和标准差，以便对数据进行归一化处理。
    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # 选取特定索引的样本数据。
        data_numpy = self.data[index]
        data_numpy = np.array(data_numpy)
        # 计算有效帧的数量。有效帧是指在所有维度上的求和不为零的帧。
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        # 调整样本数据，以适应模型的输入要求。（看一下tools里面的valid_crop_resize）
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        # 如果启用随机旋转，对数据应用随机旋转变换。
        # 为什么要做随机旋转
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        # 如果处理骨骼数据，计算骨骼之间的相对位置。
        if self.bone:
            # 每个元组(v1, v2)代表一对关节，其中v1和v2是关节的索引。
            zeb_pairs = ((1, 2), (2, 3), (3, 4), (4, 5), (5, 2), (1, 6), (6, 9), (9, 8),
                         (8, 7), (7, 6), (4, 10), (8, 10), (10, 11), (11, 12), (12, 13),
                         (14, 13), (15, 14), (16, 15), (17, 16), (17, 18), (19, 18))
            #  创建一个与原始数据形状相同的空数组，用于存储骨骼数据。
            bone_data_numpy = np.zeros_like(data_numpy)
            # 遍历骨骼点对，计算每对骨骼点之间的相对位置。
            for v1, v2 in zeb_pairs:
                # -1是得到正确的索引位置？
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            # 原始的 data_numpy 变量就被更新为处理后的骨骼数据。
            data_numpy = bone_data_numpy
        # 如果处理速度信息，计算帧与帧之间的差分。
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, index