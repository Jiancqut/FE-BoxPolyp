# from re import U
# from tkinter import image_names
# from turtle import distance, forward, shape
# from numpy import dtype, negative, positive
import torch
from torch.nn import functional as F
from torch import nn

# from adet.utils.comm import compute_locations, aligned_bilinear

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

from .custom_transforms import *

import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

def change_label(label):
    if label == 0:
        return 'Backgound'
    elif label == 1:
        return 'Poly'

def plot_embedding(data, label):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    X = (data - x_min) / (x_max - x_min)
    y = label
    fig = plt.figure(dpi=400)
    # 4-->类别个数
    for k in range(0, 4):
        print(len(np.where(y==k)[0]))
    for i, j in enumerate(np.unique(y)):
        plt.scatter(X[y == j, 0], X[y == j, 1], c = ListedColormap(('red', 'blue', 'orange', 'green'))(i), label = change_label(j), marker='*', linewidths=0.5)
    plt.xlim((0, 1.0))
    plt.ylim((0, 1.0))
    plt.legend()

    return fig

def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x

def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    # from adet.modeling.condinst.condinst import unfold_wo_center
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]


def compute_pairwise_term_new(mask_logits, pairwise_size, pairwise_dilation, images_lab, im_gt_inst):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    from adet.modeling.condinst.condinst import unfold_wo_center
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    # log_same_prob = torch.log(
    #     torch.exp(log_same_fg_prob - max_) +
    #     torch.exp(log_same_bg_prob - max_)
    # ) + max_

    # loss = -log(prob)
    # return -log_same_prob[:, 0]

    images_color_similarity = get_images_color_similarity(
        images_lab, im_gt_inst,
        pairwise_size, pairwise_dilation
    )

    log_fg_prob_unfold_outer = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation+2
    )
    log_bg_prob_unfold_outer = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation+2
    )

    log_same_fg_prob_outer = log_fg_prob[:, :, None] + log_fg_prob_unfold_outer
    log_same_bg_prob_outer = log_bg_prob[:, :, None] + log_bg_prob_unfold_outer

    max_outer = torch.max(log_same_fg_prob_outer, log_same_bg_prob_outer)

    # max_all = torch.max(max_, max_outer)

    # log_same_prob_mix = torch.log(
    #     torch.exp(log_same_fg_prob - max_) +
    #     torch.exp(log_same_bg_prob - max_)
    # ) + torch.log(
    #     torch.exp(log_same_fg_prob_outer - max_outer) +
    #     torch.exp(log_same_bg_prob_outer - max_outer)
    #     )
    # + max_ + max_outer

    # return log_same_prob
    # return log_same_prob_mix


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
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)  # connectivity参数的默认值为8
    stats = stats[stats[:, 4].argsort()]
    stats = stats[stats[:,2] > 10]

    # for b in stats[:-1]:
    #     x0, y0 = b[0], b[1]
    #     x1 = b[0] + b[2]
    #     y1 = b[1] + b[3]
    #     # print(f'x0:{x0}, y0:{y0}, x1:{x1}, y1:{y1}')
    #     start_point, end_point = (x0, y0), (x1, y1)
    #     color = (0, 0, 255) # Red color in BGR；红色：rgb(255,0,0)
    #     thickness = 2 # Line thickness of 1 px 
    #     mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # 转换为3通道图，使得color能够显示红色。
    #     mask_bboxs = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)
    
    #     cv2.imwrite('mask_bboxs.png', mask_bboxs)


    return stats[:-1]  # 排除最外层的连通图


def add_bitmasks_from_boxes(instances, images, im_h, im_w, multi_dalition=None):
    '''
    BoxInst:
    input:
    instances: gt_instances
    images: original_images.tensor
    image_masks: original_image_masks.tensor
    im_h: original_images.tensor.size(-2)
    im_w: original_images.tensor.size(-1)

    output:

    '''

    # -----> Different from add_bitmasks()
    mask_out_stride = 4
    pairwise_size = 3
    pairwise_dilation = 2

    stride = mask_out_stride
    start = int(stride // 2)

    assert images.size(2) % stride == 0
    assert images.size(3) % stride == 0

    # TODO: Why apply downsample, to adapt to the F_mask?
    downsampled_images = images.float().byte().permute(0, 2, 3, 1).cpu().numpy()
    # downsampled_images = F.avg_pool2d(
    #     images.float(), kernel_size=stride,
    #     stride=stride, padding=0
    # )[:, [2, 1, 0]]
    # image_masks = image_masks[:, start::stride, start::stride]

    # <------ End
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

        images_bitmask_full.append(torch.stack(
            per_im_bitmasks_full, dim=0))

        # 增加原始图像Lab空间值
        images_lab_collect.append(torch.cat([
            images_lab for _ in range(len(per_im_boxes))
        ], dim=0))

        images_id.extend([im_i for _ in range(len(per_im_boxes))])

        images_lab_color_similarity.append(torch.cat([
            images_color_similarity for _ in range(len(per_im_boxes))
        ], dim=0))


        if multi_dalition is not None:
            multi_images_color_similarity = get_images_color_similarity(
            images_lab, per_im_gt_inst[0],
            pairwise_size, multi_dalition
            )
            multi_images_lab_color_similarity.append(torch.cat([
            multi_images_color_similarity for _ in range(len(per_im_boxes))
            ], dim=0))
            
    if multi_dalition is None:
        return images_id, images_lab_collect, images_lab_color_similarity, images_bitmask_full, per_im_boxes_abs
    else:
        return images_id, images_lab_collect, images_lab_color_similarity, images_bitmask_full, per_im_boxes_abs, multi_images_lab_color_similarity

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
    # assert pixel_vector.size(0) == 1
    pixel_lab_size = pixel_lab.size()

    unfolded_images = unfold_wo_center(
        pixel_lab, kernel_size=kernel_size, dilation=dilation
    )

    # 距离计算1
    diff = pixel_lab[:, :, None] - unfolded_images
    diff_norm = torch.norm(diff, dim=1)

    # 距离计算2
    # diff_norm = rever_cosine_distance(pixel_vector[:, :, None], unfolded_images)

    similarity = torch.exp(-1.0 * diff_norm * alpha_3)

    return similarity

def get_vector_neighbor_diff(pixel_vector, kernel_size, dilation):

    assert pixel_vector.dim() == 4

    pixel_vector_size = pixel_vector.size()

    unfolded_images = unfold_wo_center(
        pixel_vector, kernel_size=kernel_size, dilation=dilation)

    # 差值运算
    diff = pixel_vector[:, :, None] - unfolded_images

    W = torch.norm(diff, dim=1)

    return W


def rever_cosine_distance(x, y):
    # 需要添加负号 因为相似度越高，值越大，但是作为距离函数应该要越小
    # TODO 改为 1-F.cosine_similarity(x, y)
    return 1-F.cosine_similarity(x, y)


def l2_distance(x, y):

    return torch.norm(x-y, p=2, dim=1)
    # return torch.square(x - y)


def iou_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    # Old mistake union
    # union = (x + target).sum(dim=1) + eps
    # modilfy union
    union = ((x + target - x * target).sum(dim=1) + eps)
    loss = 1. - (intersection / union)
    return loss


def compute_distance(x, y, dis_type='l2'):

    if dis_type == 'l2':
        dis = l2_distance(x, y)
    elif dis_type == 'cosine':
        dis = rever_cosine_distance(x, y)

    return dis


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=8):
        super(ASPP, self).__init__()
        modules = []
        # 注释 1
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 注释 2
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 注释 3
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 注释 4
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels,
                      out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
            # nn.Dropout(0.5)
        )

    # 注释 5
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.project(res)
        # res = F.normalize(res, dim=(2,3))
        return res


class mask_feat_attention(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(mask_feat_attention, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class consist_embedding_from_transform(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        通过进行图片的变换，达到Embedding一致性
        '''

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
        # image = self.random_gaussian_blur(image)

        image = F.normalize(image, dim=(2,3))
        
        return image

    def forward(self, images, images_embedding, embedder):

        # 获取变换后图片
        images_trans = self.img_transforms(images)

        # 获取变换后图片的Embedding
        images_trans_embedding = embedder(images_trans)

        # 将原始图片Embedding与变换后的图片Embedding做点乘，计算Embedding相似度，尽可能拉进
        loss = self.criterion(images_embedding, images_trans_embedding, self.target)

        return loss


class ContrastiveCorrelationLoss(nn.Module):

    def __init__(self):
        super(ContrastiveCorrelationLoss, self).__init__()

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def tensor_correlation(self, a, b):
        return torch.einsum("nchw,ncij->nhwij", a, b)

    def norm(self, t):
        return F.normalize(t, dim=1, eps=1e-10)
    
    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():

            fd = self.tensor_correlation(self.norm(f1), self.norm(f2))

            old_mean = fd.mean()
            fd -= fd.mean([3, 4], keepdim=True)
            fd = fd - fd.mean() + old_mean

        cd = self.tensor_correlation(self.norm(c1), self.norm(c2))
        
        min_val = 0.0

        loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def forward(self, images, images_emb, b=0.45):

        simmlarity_indx = [i for i in range(len(images))]
        random.shuffle(simmlarity_indx)

        images_sim = images[simmlarity_indx]
        images_emb_sim = images_emb[simmlarity_indx]
        
        pos_intra_loss, pos_intra_cd = self.helper(images, images_sim, images_emb, images_emb_sim, b)

        return pos_intra_loss, pos_intra_cd


class Compute_fg_bg_similarity(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        '''
        mask_feats: mask branch分支的特征图用于表示原图特征
        mask_logits: 每个Instance的预测概率
        gt_boxes: 每个Instance的boxes真值
        所有第一维度都为该batch下所有实例的个数
        '''

        self.embedding_num = 6

        # 新Embedded学习模块
        self.embedd_learned = ASPP(in_channel, (2,), self.embedding_num).to('cuda:0')
        # self.aspp_pool = mask_feat_attention(in_channels=16, out_channels=3).to('cuda:0')

        # 第三个点全局 Embedding 一致性
        self.cosist_embedding = consist_embedding_from_transform()
        self.contrasCorrelaLoss = ContrastiveCorrelationLoss()

        # 余弦距离
        # margin 默认为 1
        self.margin = 1.0

        self.aphla_1 = 1.5

        self.aphla_2 = 1.5

        self.alpha_3 = 1

        self.outer_GM_cluster_num = 5

        self.fg_GM_cluster_num = 5

        self.fg_prob_thresh = 0.85

        self.vetorch_thresh = 0.1

        self.image_id = 0
        self.last_inds = -1

        self.minMax_scaler = MinMaxScaler()

    def forward(self, mask_feats, mask_logits, gt_boxes, num_insts, gt_bitmasks, alpha=None, beta=None, warmup_factor=None, mask_out_stride=4, img_H=None, img_W=None, gt_inds=None, images_lab=None, is_training=True, images_name=None, im_inds=None, gts = None):

        assert mask_logits.dim() == 4

        fg_prob = torch.sigmoid(mask_logits)

        # 转换数据类型
        gt_boxes = torch.as_tensor(np.array(gt_boxes), device='cuda', dtype=torch.int32)
        gt_bitmasks = torch.cat(gt_bitmasks).unsqueeze(1)
        images_lab = torch.cat(images_lab)
        gt_inds = np.array(gt_inds)

        # outer_bg = None

        outer_threshold = 10

        eps = 1e-5

        # 方法二
        bs, _, _, _ = gt_bitmasks.shape
        H = img_H
        W = img_W

        ###############################################################
        # 提取边框外部分mask_feats
        # TODO: 获取boxes外部部分的mask_feats

        gt_boxes_xmin = gt_boxes[:, 0]
        gt_boxes_ymin = gt_boxes[:, 1]
        gt_boxes_xmax = gt_boxes[:, 2]
        gt_boxes_ymax = gt_boxes[:, 3]

        gt_boxes_xmin_new = gt_boxes_xmin - outer_threshold
        gt_boxes_ymin_new = gt_boxes_ymin - outer_threshold

        gt_boxes_xmax_new = gt_boxes_xmax + outer_threshold
        gt_boxes_ymax_new = gt_boxes_ymax + outer_threshold

        # 判断越界情况
        gt_boxes_xmin_new[gt_boxes_xmin_new < 0] = 0
        gt_boxes_ymin_new[gt_boxes_ymin_new < 0] = 0

        # 正确 设置
        gt_boxes_xmax_new[gt_boxes_xmax_new > W] = W
        gt_boxes_ymax_new[gt_boxes_ymax_new > H] = H

        # 提取框外背景
        outer_bg_mask = torch.zeros(bs, 1, H, W, device='cuda')
        inner_bg_mask = gt_bitmasks == 0
        # inner_bg_mask = torch.ones(bs, 1, H, W, device='cuda')

        for i in range(bs):
            outer_bg_mask[i, 0, int(gt_boxes_ymin_new[i]): int(
                gt_boxes_ymax_new[i])+1, int(gt_boxes_xmin_new[i]): int(gt_boxes_xmax_new[i])+1] = 1
            # inner_bg_mask[i, 0, gt_boxes_xmin[i]: gt_boxes_xmax[i]+1, gt_boxes_ymin[i]: gt_boxes_ymax[i]+1] = 0

        # 方法二

        # 适配 pixel vector 维度
        outer_bg_boxes_index = outer_bg_mask * inner_bg_mask

        outer_bg_boxes_index_expend = outer_bg_boxes_index.expand(-1, self.embedding_num, -1, -1)

        
        # 特征表示学习模块方法汇总
        # 方法3
        # -------------------------------------------------------

        # conv1_feat = F.relu(self.conv_1(mask_feats))
        # pixel_vector = self.conv_2(conv1_feat)
        # pixel_vector = F.normalize(pixel_vector, dim=1)

        # 新特征学习模块
        # pixel_vector = self.embedd_learned(mask_feats)

        # 新特征学习模块,增加image lab
        # images_lab = F.normalize(images_lab, dim=(2,3))
        # mask_feats_with_lab = torch.cat([mask_feats, images_lab], dim=1)
        # pixel_vector = self.embedd_learned(mask_feats_with_lab)

        # 只用lab空间值作为Embedded
        # pixel_vector = F.normalize(images_lab, dim=(2,3))

        # 只用lab空间学习Embedded
        mask_feats_with_lab = F.normalize(images_lab, dim=(2,3))
        pixel_vector = self.embedd_learned(mask_feats_with_lab)

        # 融合lab空间与 mask_feat
        # images_lab_norm = F.normalize(images_lab, dim=(2, 3))
        # mask_feats_attention = F.normalize(
        #     self.aspp_pool(mask_feats), dim=(2, 3))
        # mask_feats_with_lab = torch.cat(
        #     (images_lab_norm, mask_feats_attention), dim=1)
        # pixel_vector = self.embedd_learned(mask_feats_with_lab)

        # 第一个 loss 项
        W_ij = get_lab_neighbor_diff(
            images_lab, gt_bitmasks, 3, 1, self.alpha_3)
        vector_ij = get_vector_neighbor_diff(pixel_vector, 3, 1)

        neighbor_loss = torch.mean((W_ij >= self.vetorch_thresh) * vector_ij)

        # 第二个loss 项
        # 新求f_B的方法
        f_B_collect = []

        for gt_i in range(max(gt_inds)+1):
            if gt_i not in set(gt_inds):
                f_B_collect.append(None)
            else:
                # Fixed Bug 正确计算
                f_B_vector = pixel_vector[gt_inds == gt_i][0][None, :].masked_select((outer_bg_boxes_index == 1)[
                                                                                     gt_inds == gt_i][0][None, :]).reshape(self.embedding_num, -1).transpose(1, 0)

                if len(f_B_vector) != 0:
                    # gm_bg = GaussianMixture(n_components=self.outer_GM_cluster_num, random_state=0).fit(
                    #     f_B_vector.clone().detach().cpu().numpy())
                    kmeans = faiss.Kmeans(self.embedding_num, self.outer_GM_cluster_num)
                    kmeans.train(np.ascontiguousarray(f_B_vector.clone().detach().cpu().numpy()))
                    f_B_collect.append(torch.as_tensor(kmeans.centroids, device='cuda'))

                else:
                    f_B_collect.append(torch.zeros(
                        self.outer_GM_cluster_num, self.embedding_num).cuda())

        inner_fg_prob = gt_bitmasks * fg_prob[gt_inds]

        f_F_collect = []

        for gt_i in range(max(gt_inds)+1):
            if gt_i not in set(gt_inds):
                f_F_collect.append(None)
            else:
                bs_inner_fg_prob = torch.mean(
                    inner_fg_prob[gt_inds == gt_i], (0))[None, :]
                bigger_thresh_inds = bs_inner_fg_prob >= self.fg_prob_thresh
                pixel_vector_slice = pixel_vector[gt_inds == gt_i][0][None, :]
                # 旧 f_F_collect
                # f_F_collect.append((torch.sum(pixel_vector_slice * bigger_thresh_inds, (2,3)) / (torch.sum(bigger_thresh_inds, (2,3))+ eps)).clone().detach())

                # GM
                f_F_vector = pixel_vector_slice.masked_select(
                    bigger_thresh_inds).reshape(self.embedding_num, -1).transpose(1, 0)
                if len(f_F_vector) != 0:
                    if len(f_F_vector) == 1:
                        f_F_collect.append((torch.sum(pixel_vector_slice * bigger_thresh_inds, (2, 3)) / (
                            torch.sum(bigger_thresh_inds, (2, 3)) + eps)).clone().detach())
                    elif len(f_F_vector) < self.fg_GM_cluster_num:
                        # gm_fg = GaussianMixture(n_components=len(f_F_vector), random_state=0).fit(
                        #     f_F_vector.clone().detach().cpu().numpy())
                        kmeans = faiss.Kmeans(self.embedding_num, len(f_F_vector))
                        kmeans.train(np.ascontiguousarray(f_F_vector.clone().detach().cpu().numpy()))
                        f_F_collect.append(
                            torch.as_tensor(kmeans.centroids, device='cuda'))
                    else:
                        # gm_fg = GaussianMixture(n_components=self.fg_GM_cluster_num, random_state=0).fit(
                        #     f_F_vector.clone().detach().cpu().numpy())
                        kmeans = faiss.Kmeans(self.embedding_num, self.fg_GM_cluster_num)
                        kmeans.train(np.ascontiguousarray(f_F_vector.clone().detach().cpu().numpy()))
                        f_F_collect.append(
                            torch.as_tensor(kmeans.centroids, device='cuda'))
                else:
                    f_F_collect.append(torch.zeros(
                        self.fg_GM_cluster_num, self.embedding_num, device='cuda'))


        f_i = pixel_vector * gt_bitmasks


        ########### 第三个点 全局 Embedding 一致性 ###############
        cos_embedding = self.cosist_embedding(images_lab, pixel_vector, self.embedd_learned)
        # contrasLoss, contras_cd = self.contrasCorrelaLoss(images_lab, pixel_vector)

        ###############################

        # 第一种求均值方法
        outer_loss = []
        BoxIn_dis_B = []
        BoxIn_dis_F = []

        for i in range(bs):

            inner_mask = gt_bitmasks[i] != 0
            if torch.sum(inner_mask) == 0:
                continue
            prob_fg_choss = fg_prob[gt_inds[i]].masked_select(inner_mask).reshape(1, -1)
            f_i_i = f_i[i].masked_select(inner_mask).reshape(
                1, self.embedding_num, -1)

            dis_B = torch.norm(
                f_B_collect[gt_inds[i]].unsqueeze(2) - f_i_i, p=2, dim=1)
            dis_B_min, dis_B_index = torch.min(dis_B, dim=0)

            W_B = torch.exp(-1 * dis_B_min[None, :] * self.aphla_1)
            Prob_B = torch.square(prob_fg_choss)

            BoxIn_dis_B.append(dis_B_min)

            # 新 f_F 优化计算后
            dis_F = torch.norm(
                f_F_collect[gt_inds[i]].unsqueeze(2) - f_i_i, p=2, dim=1)
            dis_F_min, dis_F_index = torch.min(dis_F, dim=0)
            W_F = torch.exp(-1.0 * dis_F_min * self.aphla_2)

            Prob_F = torch.square(prob_fg_choss - 1.0)
            BoxIn_dis_F.append(dis_F)

            # outer_loss.append(torch.sum(W_B*Prob_B) + torch.sum(W_F*Prob_F))
            outer_loss.append(torch.mean(W_B*Prob_B) + torch.mean(W_F*Prob_F))

            is_training = True
            if not is_training:

                if gt_inds[i] == self.last_inds:
                    continue
                self.last_inds = gt_inds[i]

                tsne = TSNE(n_components=2, init='pca', random_state=2022)
                labels = gts[gt_inds[i]][0].masked_select(inner_mask).reshape(-1).clone().detach().cpu().numpy()
                result = tsne.fit_transform(f_i_i[0].transpose(0,1).clone().detach().cpu().numpy())
                fig = plot_embedding(result, labels)

                image_home = '/home/ubuntu/Projects/MA-Unet/results'
                experiment_name = 'Attention_Unet_semi_addPairwise_simli01_cosist_embedding_viewVector_tsne_new'
                experiment_name = experiment_name + '_vector'

                image_path = os.path.join(image_home, experiment_name)
                if not os.path.exists(image_path):
                    os.makedirs(image_path)

                image_B_name = '{}_{}_{}.png'.format(images_name[im_inds[i]], str(self.image_id), i)
                plt.savefig(os.path.join(image_path, image_B_name))

                continue

                prob_H, prob_W = fg_prob[gt_inds[i]][0].shape
                image_vector = torch.zeros(prob_H*prob_W, 3, dtype=torch.int64)

                dis_B_full = torch.norm(f_B_collect[gt_inds[i]].unsqueeze(
                    2) - pixel_vector[i].reshape(1, 6, -1), p=2, dim=1)
                dis_B_full_min, dis_B_full_index = torch.min(dis_B_full, dim=0)
                dis_B_full_index += 1

                dis_F_full = torch.norm(f_F_collect[gt_inds[i]].unsqueeze(
                    2) - pixel_vector[i].reshape(1, 6, -1), p=2, dim=1)
                dis_F_full_min, dis_F_full_index = torch.min(dis_F_full, dim=0)
                dis_F_full_index += 1

                image_vector[:, 0][dis_B_full_min <=
                                   dis_F_full_min] = dis_B_full_index[dis_B_full_min <= dis_F_full_min].cpu()

                image_vector[:, 1][dis_B_full_min >
                                   dis_F_full_min] = dis_F_full_index[dis_B_full_min > dis_F_full_min].cpu()

                image_vector[:, :2] = torch.ceil(
                    image_vector[:, :2]/image_vector[:, :2].max(dim=0)[0] * 255)
                image_vector = image_vector.reshape(
                    prob_H, prob_W, 3).numpy().astype(np.int32)

                # image_vector = np.ascontiguousarray(image_vector)

                x_min = int(gt_boxes_xmin[i].cpu().numpy() / img_W * prob_W)
                y_min = int(gt_boxes_ymin[i].cpu().numpy() / img_H * prob_H)

                x_max = int(gt_boxes_xmax[i].cpu().numpy() / img_W * prob_W)
                y_max = int(gt_boxes_ymax[i].cpu().numpy() / img_H * prob_H)

                image_vector_box = cv2.rectangle(
                    image_vector, (x_min, y_min, x_max, y_max), (0, 0, 255), 1)
                # image_B = torch.zeros_like(fg_prob[i], dtype=torch.int64)
                # image_B = image_B.expand()
                # image_B[inner_mask] = dis_B_index
                # image_B = torch.ceil(image_B/image_B.max() * 255).squeeze(0).unsqueeze(2)

                # image_F = torch.zeros_like(fg_prob[i], dtype=torch.int64)
                # image_F[inner_mask] = dis_F_index
                # image_F = torch.ceil(image_F/image_F.max() * 255).squeeze(0).unsqueeze(2)

                image_home = '/home/ubuntu/Projects/MA-Unet/results/Attention_Unet_semi_addPairwise_simli01_retrain_alldata_testVal_viewVector'
                experiment_name = 'Attention_Unet_semi_addPairwise_simli01_retrain_alldata_testVal_viewVector'
                experiment_name = experiment_name + '_vector'

                image_path = os.path.join(image_home, experiment_name)
                if not os.path.exists(image_path):
                    os.makedirs(image_path)

                image_B_name = '{}_{}_{}.png'.format(images_name[im_inds[i]], str(self.image_id), i)
                cv2.imwrite(os.path.join(image_path, image_B_name),
                            image_vector_box)

                # img_name = 'mask_feat_{}_{}_{}.png'.format(
                #     images_name[im_inds[i]], str(self.image_id), i)
                # fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                # for channel_id in range(16):
                #     featmap = mask_feats[i, channel_id, :,
                #                          :].clone().detach().cpu().numpy()
                #     featmap_scale = self.minMax_scaler.fit_transform(featmap)

                #     axes[int(channel_id/4), int(channel_id % 4)].imshow(np.transpose(
                #         np.expand_dims(featmap_scale, axis=0), axes=(1, 2, 0)), cmap='jet')  # im是要显示的图像
                # plt.savefig("{}/{}".format(image_path, img_name))

                # image_F_name = 'image_F_{}_{}.png'.format(str(self.image_id), i)
                # cv2.imwrite(os.path.join(image_path, image_F_name), image_F.cpu().numpy())

        self.image_id += 1

        # outer_bg_loss_mean = torch.mean(torch.hstack(BoxIn_B)) + torch.mean(torch.hstack(BoxIn_F))
        outer_bg_loss_mean = torch.mean(torch.vstack(outer_loss))
        # outer_bg_loss = torch.exp(-torch.square(f_B.unsqueeze(2) - f_i.reshape(bs, self.embedding_num, -1)) * 0.5) * torch.square(fg_prob.reshape(bs, 1, -1)) + torch.exp(-torch.square(f_F.unsqueeze(2) - f_i.reshape(bs, self.embedding_num, -1)) * 0.5 ) * torch.square(fg_prob.reshape(bs, 1, -1) - 1)
        # outer_bg_loss_mean = torch.mean(outer_bg_loss)

        # 第二种方法
        # W_B = torch.exp(-1 * compute_distance(f_B.unsqueeze(2), f_i.reshape(bs, self.embedding_num, -1)) * 0.5)
        # Prob_B = torch.square(fg_prob.reshape(bs, 1, -1))
        # BoxIn_B = torch.mean(W_B * Prob_B)

        # W_F = torch.exp(-1 * compute_distance(f_F.unsqueeze(2), f_i.reshape(bs, self.embedding_num, -1)) * 0.5)
        # Prob_F = torch.square(fg_prob.reshape(bs, 1, -1) - 1.0)
        # BoxIn_F = torch.mean(W_F * Prob_F)

        # outer_bg_loss_mean = BoxIn_B + BoxIn_F

        ###############################################################

        # value = alpha
        # storage = get_event_storage()

        # --> proj term loss

        # iou proj
        gt_bitmasks_merge = torch.cat([torch.sum(gt_bitmasks[gt_inds == im_i], dim=0) for im_i in set(gt_inds)]).unsqueeze(1)
        gt_bitmasks_merge[gt_bitmasks_merge > 1] = 1
        proj_loss = compute_new_project_term(fg_prob, gt_bitmasks_merge)

        # return proj_loss + outer_bg_loss * warmup_factor + loss_pairwise * warmup_factor
        # return proj_loss + outer_bg_loss_mean
        # return proj_loss + outer_bg_loss_mean * warmup_factor * 0.001

        # return outer_bg_loss_mean
        # return proj_loss + outer_bg_loss_mean * 10

        # return proj_loss + outer_bg_loss_mean * warmup_factor * 0.001

        # 用此返回参数
        # return proj_loss + outer_bg_loss_mean * warmup_factor

        return proj_loss, outer_bg_loss_mean, neighbor_loss, cos_embedding