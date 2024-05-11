from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

class JaccardLoss(nn.Module):
    def __init__(self, size_average=True, reduce=True, smooth=1.0):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
        self.reduce = reduce

    def jaccard_loss(self, pred, target):
        loss = 0
        for index in range(pred.size()[0]):
            iflat = pred[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()
            loss += 1 - ((intersection + self.smooth) / 
                    ( iflat.sum() + tflat.sum() - intersection + self.smooth))
        return loss / float(pred.size()[0])

    def jaccard_loss_batch(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        loss = 1 - ((intersection + self.smooth) / 
               ( iflat.sum() + tflat.sum() - intersection + self.smooth))
        return loss

    def forward(self, pred, target):
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as pred size ({})".format(target.size(), pred.size()))
        if self.reduce:
            loss = self.jaccard_loss(pred, target)
        else:    
            loss = self.jaccard_loss_batch(pred, target)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, size_average=True, reduce=True, smooth=100.0, power=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduce = reduce
        self.power = power

    def dice_loss(self, pred, target):
        loss = 0.

        for index in range(pred.size()[0]):
            iflat = pred[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()
            if self.power==1:
                loss += 1 - ((2. * intersection + self.smooth) / 
                        ( iflat.sum() + tflat.sum() + self.smooth))
            else:
                loss += 1 - ((2. * intersection + self.smooth) / 
                        ( (iflat**self.power).sum() + (tflat**self.power).sum() + self.smooth))

        return loss / float(pred.size()[0])

    def dice_loss_batch(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        
        if self.power==1:
            loss = 1 - ((2. * intersection + self.smooth) / 
                   (iflat.sum() + tflat.sum() + self.smooth))
        else:
            loss = 1 - ((2. * intersection + self.smooth) / 
                   ( (iflat**self.power).sum() + (tflat**self.power).sum() + self.smooth))
        return loss

    def forward(self, pred, target):
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as pred size ({})".format(target.size(), pred.size()))

        if self.reduce:
            loss = self.dice_loss(pred, target)
        else:    
            loss = self.dice_loss_batch(pred, target)
        return loss

class WeightedMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def weighted_mse_loss(self, pred, target, weight):
        s1 = torch.prod(torch.tensor(pred.size()[2:]).float())
        s2 = pred.size()[0]
        norm_term = (s1 * s2).cuda()
        if weight is None:
            return torch.sum((pred - target) ** 2) / norm_term
        else:
            return torch.sum(weight * (pred - target) ** 2) / norm_term

    def forward(self, pred, target, weight=None):
        return self.weighted_mse_loss(pred, target, weight)

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, pred, target, weight=None):
        return self.criterion(pred, target)


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()
    
    def forward(self, pred, target, weight=None):
        return self.criterion(pred, target)


class WeightedBCE(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, pred, target, weight=None):
        return F.binary_cross_entropy(pred, target, weight)

class WeightedCE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight_mask=None):
        loss = F.cross_entropy(pred, target, reduction='none')
        if weight_mask is not None:
            loss = loss * weight_mask
        return loss.mean()

class BinaryReg(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred):
        diff = pred - 0.5
        diff = torch.clamp(torch.abs(diff), min=1e-2)
        loss = (1.0 / diff).mean()
        return self.alpha * loss

def BCE_loss_func(output,target, weight_rate=[1,1]):
    weight = torch.FloatTensor([torch.sum(target == 1).item(), torch.sum(target == 0).item()]).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    loss = loss_fn(output, target.squeeze(1).long())
    return loss
