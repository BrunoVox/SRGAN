import torch
from tqdm import tqdm
from utils import data_parallel
from data.dataset_loaders import TestDataloader
import models.Generator as g
import os
from os import listdir, makedirs
from os.path import join, exists
from data.utils import is_image_file
from skimage.measure import compare_psnr, compare_ssim
from skimage.color import rgb2ycbcr, rgb2gray
from skimage.io import imread, imsave
import numpy as np 
from skimage import img_as_uint
from torchvision.utils import save_image

def test(model, upFactor):
    model.eval()
    ngpu = torch.cuda.device_count()
    device = torch.device('cuda' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
    
    dataset_list = [
        'Set5',
        'Set14',
        'BSD100',
        'Urban100',
        'Manga109'
    ]

    save_dict = {
        'Set5': '/home/bruno/Documents/Mestrado/SRGAN/results/SRResNet/VGG22/SRimages/Set5',
        'Set14': '/home/bruno/Documents/Mestrado/SRGAN/results/SRResNet/VGG22/SRimages/Set14',
        'BSD100': '/home/bruno/Documents/Mestrado/SRGAN/results/SRResNet/VGG22/SRimages/BSD100',
        'Urban100': '/home/bruno/Documents/Mestrado/SRGAN/results/SRResNet/VGG22/SRimages/Urban100',
        'Manga109': '/home/bruno/Documents/Mestrado/SRGAN/results/SRResNet/VGG22/SRimages/Manga109',
    }

    Set5_dir = "/home/bruno/Pictures/Set5"
    Set5_paths = [join(Set5_dir, x) for x in listdir(Set5_dir) if is_image_file(x)]
    Set14_dir = "/home/bruno/Pictures/Set14"
    Set14_paths = [join(Set14_dir, x) for x in listdir(Set14_dir) if is_image_file(x)]
    BSD100_dir = "/home/bruno/Pictures/BSD100"
    BSD100_paths = [join(BSD100_dir, x) for x in listdir(BSD100_dir) if is_image_file(x)]
    Urban100_dir = "/home/bruno/Pictures/Urban100"
    Urban100_paths = [join(Urban100_dir, x) for x in listdir(Urban100_dir) if is_image_file(x)]
    Manga109_dir = "/home/bruno/Pictures/Manga109"
    Manga109_paths = [join(Manga109_dir, x) for x in listdir(Manga109_dir) if is_image_file(x)]

    results = {
        'Set5_PSNR': [],
        'Set5_SSIM': [],
        'Set14_PSNR': [],
        'Set14_SSIM': [],
        'BSD100_PSNR': [],
        'BSD100_SSIM': [],
        'Urban100_PSNR': [],
        'Urban100_SSIM': [],
        'Manga109_PSNR': [],
        'Manga109_SSIM': [],
        'batchSize': []
    }

    testLoaders = {
        'Set5': TestDataloader(
            Set5_paths,
            upscale_factor=upFactor
        ),
        'Set14': TestDataloader(
            Set14_paths,
            upscale_factor=upFactor
        ),
        'BSD100': TestDataloader(
            BSD100_paths,
            upscale_factor=upFactor
        ),
        'Urban100': TestDataloader(
            Urban100_paths,
            upscale_factor=upFactor
        ),
        'Manga109': TestDataloader(
            Manga109_paths,
            upscale_factor=upFactor
        )
    }
    
    for dataset in dataset_list:
        testBar = tqdm(testLoaders[dataset])
        results['batchSize'] = []

        if not os.path.exists(save_dict[dataset]):
            makedirs(save_dict[dataset])

        for LRimage, HRimage, NameImg in testBar:            
            batchSize = LRimage.size(0)
            results['batchSize'].append(batchSize)

            LRimage = data_parallel(LRimage, device, ngpu)
            HRimage = data_parallel(HRimage, device, ngpu)

            with torch.set_grad_enabled(False):
                SRimage = model(LRimage)

            for i in range(batchSize):
                save_image(SRimage[i], f"{save_dict[dataset]}/SR_{(NameImg[i])[NameImg[i].rfind('/') + 1:]}", nrow=0, padding=0, normalize=True, range=(-1, 1))
                save_image(HRimage[i], f"{save_dict[dataset]}/HR_{(NameImg[i])[NameImg[i].rfind('/') + 1:]}", nrow=0, padding=0, normalize=True, range=(-1, 1))

                HR_test = rgb2ycbcr(imread(f"{save_dict[dataset]}/HR_{(NameImg[i])[NameImg[i].rfind('/') + 1:]}"))[4:-4, 4:-4, :]
                SR_test = rgb2ycbcr(imread(f"{save_dict[dataset]}/SR_{(NameImg[i])[NameImg[i].rfind('/') + 1:]}"))[4:-4, 4:-4, :]
                PSNR = compare_psnr(HR_test[:, :, 0], SR_test[:, :, 0], data_range=255)
                SSIM = compare_ssim(HR_test[:, :, 0], SR_test[:, :, 0], data_range=255, multichannel=False)
                results[f'{dataset}_PSNR'].append(PSNR)
                results[f'{dataset}_SSIM'].append(SSIM)

            testBar.set_description(
                desc=f"{dataset} PSNR -> {sum(results[f'{dataset}_PSNR']) / len(results[f'{dataset}_PSNR']):.2f} - SSIM -> {sum(results[f'{dataset}_SSIM']) / len(results[f'{dataset}_SSIM']):.4f}"
            )            

ngpu = torch.cuda.device_count()
device = torch.device('cuda' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
model = g.Generator(
    input_nc=3,
    output_nc=3,
    nf=64,
    num_resblocks=16,
    upscale_factor=4,
    norm_type='batch',
    act_type='prelu',
    init_weights=True
)
model = data_parallel(model, device, ngpu)
model.load_state_dict(torch.load('results/SRResNet/VGG22/best_gen.tar')['model_gen_state_dict'])

test(model, 4)