import torch
import numpy as np
from thop import profile
from thop import clever_format
import sys
import random
from pathlib import Path
import SimpleITK as sitk
import torch.nn as nn
import torch.nn.functional as F
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=300):
    decay =  decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))

def entropy(p):
    # compute entropy
    if (len(p.size())) == 2:
        return - torch.sum(p * torch.log(p + 1e-18)) / np.log(len(p)) / float(len(p))
    elif (len(p.size())) == 1:
        return - torch.sum(p * torch.log(p + 1e-18)) / np.log(len(p))
    else:
        raise NotImplementedError
    
def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))

#从list1中移除在list2中存在的元素，并返回移除后的结果
def remove_list(list1, list2):
    out = []
    for k in list1:
        if k in list2:
            continue
        out.append(k)
    return out

#创建目录dirname，如果目录不存在的话
def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

#将参数args的信息记录在日志文件中，其中log是一个日志对象
def log_args(args,log):
    args_info = "\n##############\n"
    for key in args.__dict__:
        args_info = args_info+(key+":").ljust(25)+str(args.__dict__[key])+"\n"
    args_info += "##############"
    log.info(args_info)

#将NumPy数组img保存为NIfTI格式的图像文件，保存路径为save_name
def save_nii(img,save_name):
    nii_image = sitk.GetImageFromArray(img)
    name = str(save_name).split("/")
    sitk.WriteImage(nii_image,str(save_name))
    print(name[-1]+" saving finished!")

#设置随机种子，用于重现实验结果
def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#根据当前的步数和总步数，计算Sigmoid函数的输出值，用于学习率的变化
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

#根据当前的步数和总步数，计算线性函数的输出值，用于学习率的变化
def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

#根据当前的步数和总步数，计算余弦函数的输出值，用于学习率的变化
def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

#根据当前的训练轮数，计算一致性损失的权重，用于自监督学习
def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=40.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

#更新指数移动平均模型的参数
def update_ema_variables(model, ema_model, alpha=0.99, global_step=0):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

#对二维图像进行中心裁剪，裁剪后的大小为det_shape。如果图像大小大于det_shape，则会进行裁剪；如果图像大小小于det_shape，则会进行填充。
def center_crop_2d(image, det_shape=[256, 256]):
    # To prevent overflow
    image = np.pad(image, ((10,10),(10,10)), mode='reflect')
    src_shape = image.shape
    shift0 = (src_shape[0] - det_shape[0]) // 2
    shift1 = (src_shape[1] - det_shape[1]) // 2
    assert shift0 > 0 or shift1 > 0, "overflow in center-crop"
    image = image[shift0:shift0+det_shape[0], shift1:shift1+det_shape[1]]
    return image

class WeightedCE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        loss = F.cross_entropy(pred, target, reduction='none')
        return loss.mean()

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()
    
    def forward(self, pred, target, weight=None):
        return self.criterion(pred, target)
