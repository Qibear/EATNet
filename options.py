import argparse

# RGBD
parser = argparse.ArgumentParser()

# TODO epoch
parser.add_argument('--epoch', type=int, default=300, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')

parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.08, help='decay rate of learning rate')

# TODO 这里将学习率衰减策略变成100
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

# TODO 预训练权重
parser.add_argument('--load', type=str, default='pre_trained/swin_base_patch4_window12_384_22k.pth',
                    help='train from checkpoints')
parser.add_argument('--load_pre', type=str, default=None,
                    help='train from checkpoints')

data_dir = "/root/autodl-tmp/Dataset/train_set"
parser.add_argument('--rgb_root', type=str, default=data_dir + "/train_images/", help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default=data_dir + "/train_depth/", help='the training depth images root')
parser.add_argument('--gt_root', type=str, default=data_dir + '/train_masks/', help='the training gt images root')
parser.add_argument('--edge_root', type=str, default=data_dir + '/train_edge/', help='the training edge images root')

data_dir_test = "/root/autodl-tmp/Dataset/test_set"
parser.add_argument('--test_rgb_root', type=str, default=data_dir_test + '/RGB/', help='the test gt images root')
parser.add_argument('--test_depth_root', type=str, default=data_dir_test + '/depth/', help='the test gt images root')
parser.add_argument('--test_gt_root', type=str, default=data_dir_test + '/GT/', help='the test gt images root')
parser.add_argument('--test_edge_root', type=str, default=data_dir_test + '/Edge/', help='the test edge images root')

# 模型存储位置
parser.add_argument('--save_path', type=str, default='./checkpoints/ckpt/', help='the path to save models and logs')
opt = parser.parse_args()
