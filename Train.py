from http import server
import yaml
import torch
import time
from attrdict import AttrDict
import os
import argparse
from datetime import datetime
import torch.nn as nn
import matplotlib.pyplot as plt
from lib.networks import Net
from lib.wealy_loss import add_bitmasks_from_boxes, Compute_fg_bg_similarity, compute_pairwise_term,embedding_loss
from utils.dataloader import get_loader,test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter, WeightedCE
import torch.nn.functional as F
import numpy as np
from torchstat import stat
import visdom
import logging
import warnings
warnings.filterwarnings("ignore")
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

env_name = 'FE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
viz = visdom.Visdom(env=env_name)
torch.backends.cudnn.benchmark = True
def structure_loss(pred, mask):
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()
@torch.no_grad()
def test(model, path, dataset):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)

    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        res1= model(image)
        res = res1 
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=True)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
 
        intersection = (input_flat*target_flat)
        
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1, num1
def train(train_loader, model, optimizer,epoch, test_path, semi_loss,scaler=None):
    model.train()
    pairwise_color_thresh = 0.1 
    pairwise_size = 3 
    pairwise_dilation = 2 
    size_rates = [1,]
    criterion = WeightedCE()
    proj_loss_record  = AvgMeter()  
    outer_bg_loss_record  = AvgMeter() 
    affinity_correlation_loss_record = AvgMeter()
    total_loss_record  = AvgMeter()   

    train_iter = 0
    for i, (images, gts, images_name) in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad(set_to_none=True)
            images = images.to('cuda:0', non_blocking=True)
            gts = gts.to('cuda:0', non_blocking=True).to(torch.int32).to(torch.float32)
            trainsize = int(round(opt.trainsize*rate/32)*32)
            train_iter += 1
            warmup_factor = min(train_iter / float(40000), 1.0)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
        
            lateral_map_1= model(images)

            im_h, im_w = images.size()[2:]

            multi_dalition = pairwise_dilation+2
            images_id, images_lab, images_lab_color_similarity, images_bitmask_full, per_im_boxes_abs, multi_images_lab_color_similarity = add_bitmasks_from_boxes(gts, images, im_h, im_w, multi_dalition)
            res = lateral_map_1
            
            res = F.interpolate(res, size=gts.shape[2:], mode='bilinear', align_corners=True)

            proj_loss, outer_bg_loss_mean = semi_loss(mask_feats=None, mask_logits=res, gt_boxes=per_im_boxes_abs, num_insts=len(per_im_boxes_abs), 
                                    gt_bitmasks=images_bitmask_full, alpha=None, beta=None, warmup_factor=None, mask_out_stride=4, 
                                    img_H=im_h, img_W=im_w, gt_inds=images_id, images_lab=images_lab, is_training=True, images_name=images_name, im_inds=images_id, gts=gts)
            pairwise_losses = compute_pairwise_term(res, pairwise_size, pairwise_dilation)
            gt_inds = np.array(images_id)
            images_lab_color_similarity_merge = torch.cat(images_lab_color_similarity)
            weights = (torch.cat([images_lab_color_similarity_merge[gt_inds == im_i][0].unsqueeze(0) for im_i in set(gt_inds)]) >= pairwise_color_thresh).float() * torch.cat([torch.sum(torch.cat(images_bitmask_full)[gt_inds == im_i], dim=0).unsqueeze(0) for im_i in set(gt_inds)]).unsqueeze(1).float()
            loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)
            multi_pairwise_losses = compute_pairwise_term(res, pairwise_size, multi_dalition)
            multi_images_lab_color_similarity_merge = torch.cat(multi_images_lab_color_similarity)
            multi_weights = (torch.cat([multi_images_lab_color_similarity_merge[gt_inds == im_i][0].unsqueeze(0) for im_i in set(gt_inds)]) >= pairwise_color_thresh).float() * torch.cat([torch.sum(torch.cat(images_bitmask_full)[gt_inds == im_i], dim=0).unsqueeze(0) for im_i in set(gt_inds)]).unsqueeze(1).float()
            multi_loss_pairwise = (multi_pairwise_losses * multi_weights).sum() / multi_weights.sum().clamp(min=1.0)
            loss_pairwise = loss_pairwise + 0.1*multi_loss_pairwise
            proj_loss = proj_loss + loss_pairwise

            embedding = F.normalize(res, p=2, dim=1)  
            embedding_target = torch.argmax(gts, dim=1)
            embedding_target[embedding_target == 0] = -1  
            mask = (embedding_target != -1).float()  
            affinity_correlation_loss = embedding_loss(embedding, embedding_target, mask, criterion)
 
            loss = proj_loss + affinity_correlation_loss +(outer_bg_loss_mean) * warmup_factor
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

    
            if rate == 1:
                proj_loss_record.update(proj_loss.data, opt.batchsize)
                outer_bg_loss_record.update(outer_bg_loss_mean.data, opt.batchsize)
                affinity_correlation_loss = affinity_correlation_loss.data
                total_loss_record.update(loss.data, opt.batchsize)
  
        if i % 20 == 0 or i == total_step or i == 1:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' [losses: {:0.4f}]'.format(datetime.now(), epoch, opt.epoch, i, total_step, total_loss_record.show())) 
            viz.line([float(proj_loss_record.show())], [(epoch-1)*total_step+train_iter], win='proj_loss', opts=dict(title='proj_loss loss'), update="append")
            viz.line([float(affinity_correlation_loss_record.show())], [(epoch-1)*total_step+train_iter], win='affinity_correlation_loss', opts=dict(title='affinity_correlation_loss loss'), update="append")
            viz.line([float(outer_bg_loss_record.show())], [(epoch-1)*total_step+train_iter], win='outer_bg_loss', opts=dict(title='outer_bg_loss loss'), update="append")
            viz.line([float(total_loss_record.show())], [(epoch-1)*total_step+train_iter], win='total_loss', opts=dict(title='total_loss loss'), update="append")
     
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)

    global best 
    if (epoch+1) % 1 == 0:
        total_dice = 0
        total_images = 0
        for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            dataset_dice, n_images = test(model, test_path, dataset)
            total_dice += (n_images*dataset_dice)
            total_images += n_images
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dict_plot[dataset].append(dataset_dice)
        meandice = total_dice/total_images
        dict_plot['test'].append(meandice)
        print('Validation dice score: {}'.format(meandice))
        logging.info('Validation dice score: {}'.format(meandice))
        if meandice > best:
            print('##################### Dice score improved from {} to {}'.format(best, meandice))
            logging.info('##################### Dice score improved from {} to {}'.format(best, meandice))
            best = meandice
            torch.save(model.state_dict(), save_path + '' + model_name + '.pth')
            torch.save(model.state_dict(), save_path +str(epoch)+ '' + model_name + '-best.pth')

if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    model_name = 'W-PolpyBox2'
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=300, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=0.0001, help='learning rate')
    parser.add_argument('--optimizer', type=str,
                        default='SGD', help='choosing optimizer Adam or SGD')
    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=200, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='/home/ubuntu/data/TrainDataset/',help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='/home/ubuntu/data/TestDataset/',help='path to testing dataset')
    parser.add_argument('--train_save', type=str,default='./model_pth/'+model_name+'/')

    opt = parser.parse_args()

    device = torch.device("cuda")
    USE_CUDA = torch.cuda.is_available()
    model = C_Net()
    model = model.to(device)

    semi_loss = Compute_fg_bg_similarity(in_channel=3).to(device)

    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': semi_loss.parameters(), 'lr': 1e-3}
            ], opt.lr)
    else:
        optimizer = torch.optim.SGD([
                {'params': model.parameters()},
                {'params': semi_loss.parameters(), 'lr': 1e-3}
            ], opt.lr, weight_decay=1e-5, momentum=0.9)
        
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, augmentation = opt.augmentation)
    total_step = len(train_loader)
    print("#"*20, "Start Training", "#"*20)

    global best
    best = 0

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer,epoch, opt.test_path, semi_loss)



