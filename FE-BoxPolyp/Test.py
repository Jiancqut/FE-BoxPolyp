import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.networks import Net
from utils.dataloader import test_dataset
import imageio
import tqdm
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '../')

torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

experiment_name = 'FE'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='/home/ubuntu/snapshots/best.pth'.format(experiment_name))

for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:

    data_path = './data/TestDataset/{}'.format(_data_name)
    
    save_path = './results/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = Net()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    print('Model loaded')

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    global_step = 0

    with tqdm.tqdm(total = test_loader.size) as pbr:
        for i in range(test_loader.size):
            global_step +=1
            # print(global_step)
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res2 = model(image)
            res = res2
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=True)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            imageio.imsave(save_path+name, res)

            pbr.update(1)
