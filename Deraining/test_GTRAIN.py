# 针对 GT-RAIN 数据集特别定制的测试代码

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

parser = argparse.ArgumentParser(description='Image Deraining using Restormer')

parser.add_argument('--input_dir', default='/root/autodl-tmp/GT-RAIN/GT-RAIN_test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/root/autodl-tmp/GT-RAIN/model-results/Restormer/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/net_g_latest.pth', type=str, help='Path to weights')

args = parser.parse_args()

####### Load yaml #######
yaml_file = 'Options/GTRAIN.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = Restormer(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 8

root_dir = args.input_dir
result_dir = args.result_dir

os.makedirs(root_dir, exist_ok=True)
scene_paths = natsorted(glob(f"{root_dir}/*"))
files = []
for scene_path in scene_paths:
    scene_name = scene_path.split('/')[-1]
    out_path = os.path.join(result_dir, scene_name)
    os.makedirs(out_path, exist_ok=True)
    rainy_img_paths = natsorted(glob(scene_path + '/*R-*.png'))
    files.extend(rainy_img_paths)
psnr_in = 0
ssim_in = 0
psnr_out = 0
ssim_out = 0

with torch.no_grad():
    for file_ in tqdm(files):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(file_))/255.
        img_t = torch.from_numpy(img).permute(2,0,1)
        input_ = img_t.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        detail, base = model_restoration(input_)
        restored = detail + base

        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]

        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        filename = file_.split('/')[-1]
        scene_name = file_.split('/')[-2]
        save_name = os.path.join(result_dir, scene_name, filename)
        utils.save_img(save_name, img_as_ubyte(restored))

        gt_file = glob(root_dir + scene_name + '/*C-000.png')[0]
        gt = np.float32(utils.load_img(gt_file))/255.
        restored = np.float32(restored)

        psnr_in += peak_signal_noise_ratio(img, gt)
        ssim_in += structural_similarity(img, gt, multichannel=True, channel_axis=-1, data_range=1)
        psnr_out += peak_signal_noise_ratio(restored, gt)
        ssim_out += structural_similarity(restored, gt, multichannel=True, channel_axis=-1, data_range=1)

print(f'PSNR input: {psnr_in/len(files)}')
print(f'SSIM input: {ssim_in/len(files)}')
print(f'PSNR output: {psnr_out/len(files)}')
print(f'SSIM output: {ssim_out/len(files)}')
