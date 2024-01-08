import argparse
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict
import numpy as np
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm


def init_seed(seed):
    # torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    parser = argparse.ArgumentParser(description='Spatial Temporal Tuples Transformer')
    parser.add_argument('--work_dir', default='./work_dir/poseR82', help='the work folder for storing results')
    parser.add_argument('--config', default='./config/poseR_joint_inference.yaml', help='path to the configuration file')

    # processor
    parser.add_argument('--run_mode', default='infer', help='must be train, test or infer')
    parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')

    # feeder for inference
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--num_worker', type=int, default=8, help='the number of worker for data loader')
    parser.add_argument('--infer_feeder_args', default=dict(), help='the arguments of data loader for inference')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model_args', default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default='./work_dir/poseR82/best_model.pt', help='the weights for model testing')

    # inference settings
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for inference')

    return parser


class Processor():
    """ Processor for Skeleton-based Action Recgnition """

    def __init__(self, arg):
        self.arg = arg
        # 以下是加载模型的代码
        self.load_model()
        self.model.eval()  # 设置为评估模式

        # 加载数据，确保这里加载的是推理数据
        self.load_data()

        # 确保模型在正确的设备上
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        # 创建推理数据加载器
        self.data_loader['infer'] = DataLoader(
            dataset=Feeder(**self.arg.infer_feeder_args),  # 确保这里使用的是推理数据的参数
            batch_size=self.arg.batch_size,  # 推理时的批量大小
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
        self.print_log('Data load finished')

    def load_model(self):
        self.output_device = torch.device("cpu")
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args).to(self.output_device)

        if self.arg.weights:
            self.print_log('Load weights from {}'.format(self.arg.weights))
            weights = torch.load(self.arg.weights, map_location=self.output_device)
            self.model.load_state_dict(weights)
        self.print_log('Model load finished: ' + self.arg.model)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        # if self.arg.print_log:
        #     with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
        #         print(str, file=f)

    def infer(self, loader_name=['infer'], result_file=None):
        self.model.eval()
        result_file = 'inference_results.pkl'
        all_outputs = []
        for ln in loader_name:
            for batch, (data, sample) in enumerate(tqdm(self.data_loader[ln], desc="Inferencing", ncols=100)):
                with torch.no_grad():
                    data = data.float()  # 如果使用GPU，则添加 .cuda(self.output_device)
                    output = self.model(data)
                    all_outputs.append(output.data.cpu().numpy())  # 收集输出

            # 合并所有批次的输出
            combined_output = np.concatenate(all_outputs)

            # 处理和保存结果
            if result_file:
                result_dict = dict(zip(self.data_loader[ln].dataset.sample_name, combined_output))
                with open(f'{self.arg.work_dir}/{result_file}', 'wb') as f:
                    pickle.dump(result_dict, f)

        self.print_log('Inference completed.')

    def start(self):

        if self.arg.run_mode == 'infer':
            # 推理模式
            self.print_log('Model: {}'.format(self.arg.model))
            self.print_log('Starting inference...')
            self.infer(loader_name=['infer'])  # 假设您已经有了一个名为 'infer' 的数据加载器
            self.print_log('Inference completed.\n')


if __name__ == '__main__':

    os.chdir('/Users/lanxinxu/Desktop/INTERN_2023/PycharmProjects/pythonProject/zebrafish_locomotion/ZebrafishMAE')
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = arg.cuda_visible_device
    init_seed(1)
    processor = Processor(arg)
    processor.start()
