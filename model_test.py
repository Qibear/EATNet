import sys

import torch
import torch.nn.functional as F

sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.model import Detector
from tools.data import test_dataset
from tqdm import tqdm

test_data_path = "E:/Data-Center/Datatset/SOD_Dataset/TestDataset-Main/"
parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str, default=test_data_path,
                    help='test dataset path')

parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
opt = parser.parse_args()

dataset_path = opt.test_path

# set device for test
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

model = Detector()
pth_path = "checkpoints/save_model_epoch_300.pth"
model.load_state_dict(torch.load(pth_path))
model.cuda()
model.eval()

# TODO 设置测试数据集
# test_datasets = ['LFSD', 'NJU2K', 'NLPR', 'SSD', 'STERE']
test_datasets = ['SIP']
# test_datasets = ['RGBD135', 'SIP']

# TODO 遍历测试数据集，进行测试
for dataset in test_datasets:

    save_path = './detection_res_ep300/res_sod/' + dataset + '/'
    edge_save_path = './detection_res/res_edge/' + dataset + '/'

    # 创建存储文件夹
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(edge_save_path):
        os.makedirs(edge_save_path)

    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/depth/'

    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)

    # 开始测试
    for i in tqdm(range(test_loader.size), postfix=dataset):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image = image.cuda()
        depth = depth = depth.repeat(1, 3, 1, 1).cuda()
        res, edge, res1 = model(image, depth)

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        edge = F.upsample(edge, size=gt.shape, mode='bilinear', align_corners=False)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        edge = edge.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)

        # 存储
        # print('save img to: ', save_path + name)
        cv2.imwrite(save_path + name, res * 255)
        cv2.imwrite(edge_save_path + name, edge * 255)

print('Test Done!')
