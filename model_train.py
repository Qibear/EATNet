import os
import torch
import torch.nn.functional as F
import sys

sys.path.append('./models')

import numpy as np
from datetime import datetime
from models.model import Detector
from torchvision.utils import make_grid
from tools.data import get_loader, test_dataset
from tools.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    np.random.seed(seed)  # 修改
    torch.backends.cudnn.deterministic = True


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()

    loss_all = 0
    epoch_step = 0

    try:
        # 从数据载入器读取RGB图像，深度图，以及处理后的边缘器
        for i, (images, gts, depth, edge) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()
            edge = edge.cuda()
            s, e, s1 = model(images, depth)
            sal_loss = CE(s, gts)
            sal_loss1 = CE(s1, gts)

            edge_loss = CE(e, edge)
            loss = sal_loss + edge_loss + sal_loss1
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||sal_loss:{:4f} ||edge_loss:{:4f} '
                    '||sal_loss1:{:4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           optimizer.state_dict()['param_groups'][0]['lr'], sal_loss.data, edge_loss.data,
                           sal_loss1.data))

                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f} ||edge_loss:{:4f}||sal_loss1:{:4f} , mem_use:{:.0f}MB'.
                    format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'],
                           sal_loss.data, edge_loss.data, sal_loss1.data, memory_used))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = s[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        # TODO 每隔5个epoch存储一下训练文件
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'save_model_epoch_{}.pth'.format(epoch + 1))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'save_model__epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# test function

def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()
            res, e, res1 = model(image, depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}\n'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'CSNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))  # 存储mae最小的权重文件

        # 写入数据
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    setup_seed(1998)

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    cudnn.benchmark = True
    image_root = opt.rgb_root
    gt_root = opt.gt_root
    depth_root = opt.depth_root
    edge_root = opt.edge_root

    test_image_root = opt.test_rgb_root
    test_gt_root = opt.test_gt_root
    test_depth_root = opt.test_depth_root
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # log写入
    logging.basicConfig(filename=save_path + 'RGBD.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                        datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Detector-Train_logs_pairs")

    # TODO 模型实例化
    model = Detector()
    num_parms = 0
    if opt.load is not None:
        model.load_pre(opt.load)
        print('load model from ', opt.load)

    model.cuda()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('load data...')
    train_dataloader = get_loader(image_root, gt_root, depth_root, edge_root, batchsize=opt.batchsize,
                                  trainsize=opt.trainsize)
    test_dataloader = test_dataset(test_image_root, test_gt_root, test_depth_root, opt.trainsize)
    total_step = len(train_dataloader)
    logging.info("Config")
    logging.info(
        'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
            opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
            opt.decay_epoch))
    CE = torch.nn.BCEWithLogitsLoss()
    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0

    """--------------------------------------------------------------------------"""
    print("Start train...")
    for epoch in range(1, opt.epoch):
        # if (epoch % 50 ==0 and epoch < 60):
        # cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.epoch - 50, min_lr=opt.lr * opt.decay_rate * opt.decay_rate)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_dataloader, model, optimizer, epoch, save_path)
        test(test_dataloader, model, epoch, save_path)
