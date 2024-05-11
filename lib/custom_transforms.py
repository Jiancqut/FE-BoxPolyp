import torch 
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np 
from PIL import Image, ImageFilter
import random 
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

class RandomGaussianBlur(object):
    def __init__(self, sigma, p):
        self.min_x = sigma[0]
        self.max_x = sigma[1]
        self.del_p = 1 - p
        self.p_ref = p
        self.plist = np.random.random_sample(1)

    def __call__(self, image):
        if self.plist < self.p_ref:
            x = self.plist - self.p_ref 
            m = (self.max_x - self.min_x) / self.del_p
            b = self.min_x 
            s = m * x + b 

            return image.filter(ImageFilter.GaussianBlur(radius=s))
        else:
            return image

class RandomGrayScale(object):
    def __init__(self, p):
        self.grayscale = transforms.RandomGrayscale(p=1.) # Deterministic (We still want flexible out_dim).
        self.p_ref = p
        self.plist = np.random.random_sample(1)

    def __call__(self, image):
        if self.plist < self.p_ref:
            return self.grayscale(image)
        else:
            return image


class RandomColorBrightness(object):
    def __init__(self, x, p):
        self.min_x = max(0, 1 - x)
        self.max_x = 1 + x
        self.p_ref = p
        self.plist = np.random.random_sample(1)
        self.rlist = random.uniform(self.min_x, self.max_x)

    def __call__(self, image):
        if self.plist < self.p_ref:
            return TF.adjust_brightness(image, self.rlist)
        else:
            return image


class RandomColorContrast(object):
    def __init__(self, x, p):
        self.min_x = max(0, 1 - x)
        self.max_x = 1 + x
        self.p_ref = p
        self.plist = np.random.random_sample(1)
        self.rlist = random.uniform(self.min_x, self.max_x)

    def __call__(self, image):
        if self.plist < self.p_ref:
            return TF.adjust_contrast(image, self.rlist)
        else:
            return image


class RandomColorSaturation(object):
    def __init__(self, x, p):
        self.min_x = max(0, 1 - x)
        self.max_x = 1 + x
        self.p_ref = p
        self.plist = np.random.random_sample(1)
        self.rlist = random.uniform(self.min_x, self.max_x)

    def __call__(self, image):
        if self.plist < self.p_ref:
            return TF.adjust_saturation(image, self.rlist)
        else:
            return image


class RandomColorHue(object):
    def __init__(self, x, p):
        self.min_x = -x
        self.max_x = x
        self.p_ref = p
        self.plist = np.random.random_sample(1)
        self.rlist = random.uniform(self.min_x, self.max_x)

    def __call__(self, image):
        if self.plist < self.p_ref:
            return TF.adjust_hue(image, self.rlist)
        else:
            return image
