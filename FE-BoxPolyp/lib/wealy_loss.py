import torch
from torch.nn import functional as F
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import cv2
from skimage import color
import faiss
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import os, sys
sys.path.append(os.getcwd())
from lib.custom_transforms import *
import warnings
warnings.filterwarnings("ignore")
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '../')


def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]), dim=2)
    return unfolded_x

def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_
    return -log_same_prob[:, 0]
# 计算图像颜色相似度
def get_images_color_similarity(images, image_masks, kernel_size, dilation):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]


    return similarity * unfolded_weights

def mask_find_bboxs(mask):
    mask = mask[0].cpu().numpy().astype(np.uint8)
    ret, mask = cv2.threshold(mask*255, 127, 255, cv2.THRESH_BINARY)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)  
    stats = stats[stats[:, 4].argsort()]
    stats = stats[stats[:,2] > 10]

    for b in stats[:-1]:
         x0, y0 = b[0], b[1]
         x1 = b[0] + b[2]
         y1 = b[1] + b[3]

         start_point, end_point = (x0, y0), (x1, y1)
         color = (0, 0, 255) 
         thickness = 2  
         mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
         mask_bboxs = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)
    
         cv2.imwrite('mask_bboxs.png', mask_bboxs)


    return stats[:-1] 

def add_bitmasks_from_boxes(instances, images, im_h, im_w, multi_dalition=None):
    mask_out_stride = 4
    pairwise_size = 3
    pairwise_dilation = 2

    stride = mask_out_stride
    start = int(stride // 2)

    assert images.size(2) % stride == 0
    assert images.size(3) % stride == 0


    downsampled_images = images.float().byte().permute(0, 2, 3, 1).cpu().numpy()


    images_id = []
    images_bitmask_full = []
    images_lab_collect = []
    images_lab_color_similarity = []
    multi_images_lab_color_similarity = []
    per_im_boxes_abs = []

    for im_i, per_im_gt_inst in enumerate(instances):
        images_lab = color.rgb2lab(
            downsampled_images[im_i])
        images_lab = torch.as_tensor(
            images_lab, device=images.device, dtype=torch.float32)
        images_lab = images_lab.permute(2, 0, 1)[None]
        images_color_similarity = get_images_color_similarity(
            images_lab, per_im_gt_inst[0],
            pairwise_size, pairwise_dilation
        )

        per_im_boxes = mask_find_bboxs(per_im_gt_inst)

        per_im_bitmasks_full = []

        for per_box in per_im_boxes:
            bitmask_full = torch.zeros((im_h, im_w), device=images.device, dtype=torch.float32)

            x_0, x_1 = int(per_box[0]), int(per_box[0] + per_box[2])
            y_0, y_1 = int(per_box[1]), int(per_box[1] + per_box[3])
            
            target_box = [x_0, y_0, x_1, y_1]

            bitmask_full[int(target_box[1]):int(target_box[3] + 1),
                         int(target_box[0]):int(target_box[2] + 1)] = 1.0

            assert bitmask_full.size(0) == im_h
            assert bitmask_full.size(1) == im_w

            per_im_bitmasks_full.append(bitmask_full)
            per_im_boxes_abs.append(target_box)
        if len(per_im_bitmasks_full) > 0:
            images_bitmask_full.append(torch.stack(per_im_bitmasks_full, dim=0))

        if len(per_im_boxes) != 0:
            images_lab_collect.append(torch.cat([images_lab for _ in range(len(per_im_boxes))], dim=0))
            images_id.extend([im_i for _ in range(len(per_im_boxes))])

        if multi_dalition is not None and len(per_im_boxes)!= 0 :
            multi_images_color_similarity = get_images_color_similarity(images_lab, per_im_gt_inst[0],pairwise_size, multi_dalition)
            multi_images_lab_color_similarity.append(torch.cat([multi_images_color_similarity for _ in range(len(per_im_boxes))], dim=0))
            
    if multi_dalition is None:
        return images_id, images_lab_collect, images_lab_color_similarity, images_bitmask_full, per_im_boxes_abs
    else:
        return images_id, images_lab_collect, images_lab_color_similarity, images_bitmask_full, per_im_boxes_abs, multi_images_lab_color_similarity

def iou_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1) # 交集
    union = ((x + target - x * target).sum(dim=1) + eps)
    loss = 1. - (intersection / union)
    return loss

def compute_new_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = iou_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = iou_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_y + mask_losses_x).mean()

def get_lab_neighbor_diff(pixel_lab, image_masks, kernel_size, dilation, alpha_3):   
    assert pixel_lab.dim() == 4

    pixel_lab_size = pixel_lab.size()

    unfolded_images = unfold_wo_center(pixel_lab, kernel_size=kernel_size, dilation=dilation)

    diff = pixel_lab[:, :, None] - unfolded_images
    diff_norm = torch.norm(diff, dim=1)

    similarity = torch.exp(-1.0 * diff_norm * alpha_3)
    return similarity

def get_vector_neighbor_diff(pixel_vector, kernel_size, dilation):
    assert pixel_vector.dim() == 4

    pixel_vector_size = pixel_vector.size()

    unfolded_images = unfold_wo_center(
        pixel_vector, kernel_size=kernel_size, dilation=dilation)

    diff = pixel_vector[:, :, None] - unfolded_images
    W = torch.norm(diff, dim=1)
    return W
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(Conv, self).__init__(*modules)
class Pooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Pooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)
class embedder(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=8):
        super(embedder, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(Conv(in_channels, out_channels, rate))

        modules.append(Pooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels,
                      out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),

        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.project(res)

        return res
class consist_embedding_from_transform(nn.Module):
    def __init__(self):
        super().__init__()
     
        self.random_color_brightness = RandomColorBrightness(x=0.3, p=0.8)
        self.random_color_contrast   = RandomColorContrast(x=0.3, p=0.8)
        self.random_color_saturation = RandomColorSaturation(x=0.3, p=0.8)
        self.random_color_hue        = RandomColorHue(x=0.1, p=0.8)
        self.random_gray_scale    = RandomGrayScale(p=0.2)
        self.random_gaussian_blur = RandomGaussianBlur(sigma=[.1, 2.], p=0.5)

        self.criterion = nn.CosineEmbeddingLoss()
        self.target = torch.ones(size=(1,), device='cuda')

    def img_transforms(self, image):

        image = self.random_color_brightness(image)
        image = self.random_color_contrast(image)
        image = self.random_color_saturation(image)
        image = self.random_color_hue(image)
        image = self.random_gray_scale(image)

        image = F.normalize(image, dim=(2,3))
        return image
    def forward(self, images, images_embedding, embedder):


        images_trans = self.img_transforms(images)


        images_trans_embedding = embedder(images_trans)
        images_embedding = images_embedding.view(-1, 8)

        images_trans_embedding = images_trans_embedding.view(-1, 8)


        loss = self.criterion(images_embedding, images_trans_embedding, self.target)

        return loss
class Compute_fg_bg_similarity(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.embedding_num = 6
        self.embedd_learned = embedder(in_channel, (2,), self.embedding_num).to('cuda:0')
        self.cosist_embedding = consist_embedding_from_transform()
        self.margin = 1.0
        self.aphla_1 = 1.5
        self.alpha_3 = 1
        self.outer_GM_cluster_num = 5
        self.fg_GM_cluster_num = 5
        self.fg_prob_thresh = 0.80
        self.vetorch_thresh = 0.15
        self.image_id = 0
        self.last_inds = -1
        self.minMax_scaler = MinMaxScaler()

    def forward(self, mask_feats, mask_logits, gt_boxes, num_insts, gt_bitmasks, alpha=None, beta=None, warmup_factor=None, mask_out_stride=4, img_H=None, img_W=None, gt_inds=None, images_lab=None, is_training=True, images_name=None, im_inds=None, gts = None):
        assert mask_logits.dim() == 4
        fg_prob = torch.sigmoid(mask_logits)
        gt_boxes = torch.as_tensor(np.array(gt_boxes), device='cuda', dtype=torch.int32)
        gt_bitmasks = torch.cat(gt_bitmasks).unsqueeze(1)
        images_lab = torch.cat(images_lab)
        gt_inds = np.array(gt_inds)
        outer_threshold = 10
        bs, _, _, _ = gt_bitmasks.shape
        H = img_H
        W = img_W

        gt_boxes_xmin = gt_boxes[:, 0]
        gt_boxes_ymin = gt_boxes[:, 1]
        gt_boxes_xmax = gt_boxes[:, 2]
        gt_boxes_ymax = gt_boxes[:, 3]

        gt_boxes_xmin_new = gt_boxes_xmin - outer_threshold
        gt_boxes_ymin_new = gt_boxes_ymin - outer_threshold

        gt_boxes_xmax_new = gt_boxes_xmax + outer_threshold
        gt_boxes_ymax_new = gt_boxes_ymax + outer_threshold


        gt_boxes_xmin_new[gt_boxes_xmin_new < 0] = 0
        gt_boxes_ymin_new[gt_boxes_ymin_new < 0] = 0


        gt_boxes_xmax_new[gt_boxes_xmax_new > W] = W
        gt_boxes_ymax_new[gt_boxes_ymax_new > H] = H


        outer_bg_mask = torch.zeros(bs, 1, H, W, device='cuda')
        inner_bg_mask = gt_bitmasks == 0

        for i in range(bs):
            outer_bg_mask[i, 0, int(gt_boxes_ymin_new[i]): int(gt_boxes_ymax_new[i])+1, int(gt_boxes_xmin_new[i]): int(gt_boxes_xmax_new[i])+1] = 1

        outer_bg_boxes_index = outer_bg_mask * inner_bg_mask
        mask_feats_with_lab = F.normalize(images_lab, dim=(2,3))
        pixel_vector = self.embedd_learned(mask_feats_with_lab)

        W_ij = get_lab_neighbor_diff(images_lab, gt_bitmasks, 3, 1, self.alpha_3)
        vector_ij = get_vector_neighbor_diff(pixel_vector, 3, 1)
        neighbor_loss = torch.mean((W_ij >= self.vetorch_thresh) * vector_ij)

        f_B_collect = []
        for gt_i in range(max(gt_inds)+1):
            if gt_i not in set(gt_inds):
                f_B_collect.append(None)
            else:
                f_B_vector = pixel_vector[gt_inds == gt_i][0][None, :].masked_select((outer_bg_boxes_index == 1)[gt_inds == gt_i][0][None, :]).reshape(self.embedding_num, -1).transpose(1, 0)
                if len(f_B_vector) != 0:
                    kmeans = faiss.Kmeans(self.embedding_num, self.outer_GM_cluster_num)
                    kmeans.train(np.ascontiguousarray(f_B_vector.clone().detach().cpu().numpy()))
                    f_B_collect.append(torch.as_tensor(kmeans.centroids, device='cuda'))
                else:
                    f_B_collect.append(torch.zeros(self.outer_GM_cluster_num, self.embedding_num).cuda())

        f_F_collect = []
        for gt_i in range(max(gt_inds)+1):
            if gt_i not in set(gt_inds):
                f_F_collect.append(None)
            else:
                f_F_vector = pixel_vector[gt_inds == gt_i][0][None, :].masked_select((outer_bg_boxes_index == 1)[
                                     gt_inds == gt_i][0][None, :]).reshape(self.embedding_num, -1).transpose(1, 0)
                if len(f_F_vector) != 0:
                    kmeans = faiss.Kmeans(self.embedding_num, self.fg_GM_cluster_num)
                    kmeans.train(np.ascontiguousarray(f_F_vector.clone().detach().cpu().numpy()))
                    f_F_collect.append(torch.as_tensor(kmeans.centroids, device='cuda'))
                else:
                    f_F_collect.append(torch.zeros(
                        self.fg_GM_cluster_num, self.embedding_num, device='cuda'))

        f_i = pixel_vector * gt_bitmasks
        cos_embedding = self.cosist_embedding(images_lab, pixel_vector, self.embedd_learned) 
        outer_loss = []
        for i in range(bs):
            inner_mask = gt_bitmasks[i] != 0
            if torch.sum(inner_mask) == 0:
                continue
            prob_fg_choss = fg_prob[gt_inds[i]].masked_select(inner_mask).reshape(1, -1)
            f_i_i = f_i[i].masked_select(inner_mask).reshape(1, self.embedding_num, -1)

            dis_B = torch.norm(f_B_collect[gt_inds[i]].unsqueeze(2) - f_i_i, p=2, dim=1)
            dis_B_min, dis_B_index = torch.min(dis_B, dim=0)
            W_B = torch.exp(-1 * dis_B_min[None, :] * self.aphla_1)
            Prob_B = torch.square(prob_fg_choss)
            dis_F = torch.norm(f_F_collect[gt_inds[i]].unsqueeze(2) - f_i_i, p=2, dim=1)
            dis_F_min, dis_F_index = torch.min(dis_F, dim=0)
            W_F = torch.exp(-1.0 * dis_F_min * self.aphla_1)
            Prob_F = torch.square(prob_fg_choss - 1.0)
            outer_loss.append(torch.mean(W_B*Prob_B) + torch.mean(W_F*Prob_F))

        self.image_id += 1
        outer_bg_loss_mean = torch.mean(torch.vstack(outer_loss))

        self.image_id += 1
        outer_bg_loss_mean = torch.mean(torch.vstack(outer_loss))

        gt_bitmasks_merge = torch.cat([torch.sum(gt_bitmasks[gt_inds == im_i], dim=0) for im_i in set(gt_inds)]).unsqueeze(1)
        gt_bitmasks_merge[gt_bitmasks_merge > 1] = 1
        proj_loss = compute_new_project_term(fg_prob, gt_bitmasks_merge)
        return proj_loss, outer_bg_loss_mean, neighbor_loss , cos_embedding    
class EmbeddingLoss:
    def __init__(self, criterion, device):
        self.criterion = criterion
        self.device = device

    def __call__(self, embedding, target, mask):
        embedding = F.normalize(embedding, p=2, dim=1)
        mask = mask.float().to(self.device)

        loss = torch.tensor(0, dtype=embedding.dtype, device=self.device)
        loss_temp = self.embedding_single_offset_loss(embedding, target, mask)
        loss += loss_temp

        return loss
    def embedding_single_offset_loss(self, embedding, target, mask):
        affs_temp = torch.sum(embedding * embedding, dim=1)
        loss_temp = self.criterion(affs_temp * mask, target * mask)
        return loss_temp

    