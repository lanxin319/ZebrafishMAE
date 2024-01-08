#!/usr/bin/env python
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
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='Spatial Temporal Tuples Transformer')
    parser.add_argument('--work_dir', default='./work_dir/zebrafish210', help='the work folder for storing results')
    parser.add_argument('--config', default='./config/zebrafish_bone.yaml', help='path to the configuration file')

    # processor
    parser.add_argument('--run_mode', default='train', help='must be train or test')
    parser.add_argument('--save_score', type=str2bool, default=False,
                        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--save_epoch', type=int, default=80, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval_interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--num_worker', type=int, default=8, help='the number of worker for data loader')
    parser.add_argument('--train_feeder_args', default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test_feeder_args', default=dict(), help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model_args', default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default='./work_dir/zebrafish210/best_model_PKU.pt', help='the weights for model testing')
    parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base_lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[60, 80], nargs='+',
                        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--cuda_visible_device', default='0,1', help='')
    parser.add_argument('--device', type=int, default=[0, 1], nargs='+',
                        help='the indexes of GPUs for training or testing')

    parser.add_argument('--optimizer', default='Adam', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=5)

    return parser


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class Processor():
    """ Processor for Skeleton-based Action Recgnition """

    def __init__(self, arg):
        self.arg = arg
        self.global_step = 0
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_loss = float('inf')  # 我加的，初始化最佳损失为无穷大

        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        # 加载模型先注释掉，现在还没有模型
        self.load_model()
        # 加载数据
        self.load_data()

        if arg.run_mode == 'train':
            result_visual = os.path.join(arg.work_dir, 'runs')
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(result_visual):
                    print('log_dir: ', result_visual, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(result_visual)
                        print('Dir removed: ', result_visual)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', result_visual)
                self.train_writer = SummaryWriter(os.path.join(result_visual, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(result_visual, 'val'), 'val')

                self.load_optimizer()
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(result_visual, 'test'), 'test')

        # self.model = self.model.cuda(self.output_device)
        self.model = self.model.cpu()
        # self.model = self.model.to('mps')

        # if type(self.arg.device) is list:
        #     if len(self.arg.device) > 1:
        #         # 数据并行，把batch_size平均分到两个GPU上训练，一个GPU放32
        #         self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.run_mode == 'train':
            self.data_loader['train'] = DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
        self.print_log('Data load finished')

    def load_model(self):
        # 用哪个GPU
        # output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        output_device = 'cpu'
        # output_device = 'mps'
        self.output_device = output_device
        # 他们是model.sttformer.Model，我要自己搭一个
        Model = import_class(self.arg.model)
        # print(Model)
        # 加载模型的一些参数
        self.model = Model(**self.arg.model_args)
        # print(self.model)
        # 这里是交叉熵因为是分类的，我要换成MSE
        # self.loss = nn.CrossEntropyLoss().cuda(output_device)
        # self.loss = nn.MSELoss(reduction='mean').cuda(output_device)
        self.loss = nn.MSELoss(reduction='mean')

        # 训练后才有的weights
        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            # weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
        self.print_log('Model load finished: ' + self.arg.model)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        self.print_log('Optimizer load finished: ' + self.arg.optimizer)

    def adjust_learning_rate(self, epoch):
        self.print_log('adjust learning rate, using warm up, epoch: {}'.format(self.arg.warm_up_epoch))
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def train(self, epoch, save_model=False):
        losses = AverageMeter()

        self.model.train()
        self.adjust_learning_rate(epoch)

        self.train_writer.add_scalar('epoch', epoch, self.global_step)

        # 这里去掉了label，sample是样本的索引。
        for batch, (data, sample) in enumerate(tqdm(self.data_loader['train'], desc="Training", ncols=100)):
            self.global_step += 1
            with torch.no_grad():
                # data = data.float().cuda(self.output_device)
                data = data.float()

            # forward
            # 之前的output等于 [batch_size, num_class]，现在修改过后跟原始数据维度一样
            output = self.model(data)
            # loss = self.loss(output, label)
            loss = self.loss(output, data)  # 原来是跟label比，现在是跟原始数据比
            # backward
            self.optimizer.zero_grad()  # 先对梯度进行一个清零
            loss.backward()  # 计算梯度
            # w' = w - lr * grad:
            self.optimizer.step()  # 更新梯度

            # 要加item()是因为loss是一个tensor类型
            # train_loss.append(loss.item())

            losses.update(loss.item())

            self.train_writer.add_scalar('loss', losses.avg, self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)

        # 完成了一次iteration：
        self.print_log('training: epoch: {}, loss: {:.4f}, lr: {:.6f}'.format(
            epoch + 1, losses.avg, self.lr))

        # 记录和比较损失，决定是否保存模型
        current_loss = losses.avg
        if current_loss < self.best_loss:
            self.best_loss = current_loss  # 更新最佳损失
            self.print_log('New best loss {:.4f}, saving model...'.format(current_loss))
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.work_dir + '/' + 'best_model_PKU.pt')  # 保存最佳模型的权重

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        losses = AverageMeter()

        self.model.eval()
        for ln in loader_name:
            score_frag = []
            for batch, (data, sample) in enumerate(tqdm(self.data_loader[ln], desc="Evaluating", ncols=100)):
                # label_list.append(label)
                with torch.no_grad():
                    # data = data.float().cuda(self.output_device)
                    data = data.float()
                    output = self.model(data)
                    loss = self.loss(output, data)

                    score_frag.append(output.data.cpu().numpy())
                    losses.update(loss.item())

            score = np.concatenate(score_frag)

            if self.arg.run_mode == 'train':
                self.val_writer.add_scalar('acc', losses.avg, self.global_step)

            self.print_log('evaluating: loss: {:.4f}'.format(losses.avg))

            if save_score:
                score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
                with open('{}/score.pkl'.format(self.arg.work_dir), 'wb') as f:
                    pickle.dump(score_dict, f)

    def start(self):

        if self.arg.run_mode == 'train':
            for argument, value in sorted(vars(self.arg).items()):
                self.print_log('{}: {}'.format(argument, value))

            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.print_log(f'# Parameters: {count_parameters(self.model)}')

            self.print_log('###***************start training***************###')

            # loss_history = []  # 存储训练过程中的损失，我加的

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (epoch + 1 == self.arg.num_epoch)
                self.train(epoch, save_model=save_model)

                if ((epoch + 1) % self.arg.eval_interval == 0):
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

                # 可选：保存损失历史记录以供以后可视化
                # loss_history.append(self.train_losses.avg)

            self.print_log('Done.\n')

        elif self.arg.run_mode == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}'.format(self.arg.model))
            self.print_log('Weights: {}'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'])
            self.print_log('Done.\n')


if __name__ == '__main__':

    # os.chdir('/home/wangshuo/Desktop/ZebrafishMAE')
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