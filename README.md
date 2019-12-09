# SRGAN
PyTorch implementation of SRGAN

This work is based on [Ledig et al.](https://arxiv.org/abs/1609.04802) It is focused in being as close as possible to the original proposition.
For training/validation, I did use a sample of aproximately 475k images from Imagenet. For testing, I used Set5, Set14, BSD100, Urban100 and Manga109.

## 1. Training dataset:

Choose a training dataset like Imagenet, COCO or DIV2K. Just make sure to train the models with aproximately a million backpropagations, as the paper suggests. For my case, it was with 40 epochs. In the future, I will code it to adapt to every dataset size. Be sure to edit /config/train.json with the path you're using for your training dataset, and make sure this path does not contain subfolders with part of the dataset.

## 2. Test datasets:

I used 5 datasets to evaluate the results. The paper only uses 3 of those, so the other 2 are extra for comparing with related works. Be sure to edit /config/test.json with the path you're using for your test datasets. This path must contain 5 folders named after the datasets.
Download the datasets in this link: [Datasets](https://drive.google.com/file/d/1ZMakMnF7XoWGiKBivsa6zrilLFxq7529/view?usp=sharing)

## 3. Train the desired model:

Run the main.py file and chose the model you want to train and the loss you want to use. Training any model takes about 16 hours in a single GTX1070.

## 4. Test the desired model:

Run the test.py file and see the PSNR and SSIM results. A series of SR images will be generated in the process. They will be in /results/"model"/"loss"/SRimages.

## 5. Optional:

SRResNet-MSE pretrained model: [SRResNet-MSE](https://drive.google.com/file/d/14jQaDPsfbKd3zOOE-xH83o2-bT1Irgje/view?usp=sharing)

SRResNet-VGG22 pretrained model: [SRResNet-VGG22](https://drive.google.com/file/d/1-ZkQiAU2wSCpvuyi0Wn6B0liXCr3KW15/view?usp=sharing)

SRGAN results yet to come...

To use the pretrained models, just paste them at /results/"model"/"loss"/

## Results (x4):

Results in brackets are reported from the authors. NR stands for "not reported".

SRResNet-MSE

| Dataset | PSNR | SSIM |
| :---         |     :---:      |   :---: |
| Set5   | 32.10 [32.05]    | 0.9035 [0.9019]   |
| Set14     | 28.76 [28.49]      | 0.8020 [0.8184]     |
| BSD100   | 28.50 [27.58]    | 0.7825 [0.7620]   |
| Urban100     | 26.11 [NR]      | 0.7991 [NR]     |
| Manga109   | 30.67 [NR]    | 0.9181 [NR]   |

SRResNet-VGG22

| Dataset | PSNR | SSIM |
| :---         |     :---:      |   :---: |
| Set5   | 29.79 [30.51]    | 0.8706 [0.8803]   |
| Set14     | 26.78 [27.19]      | 0.7481 [0.7807]     |
| BSD100   | 26.58 [NR]    | 0.7199 [NR]   |
| Urban100     | 24.89 [NR]      | 0.7583 [NR]     |
| Manga109   | 29.30 [NR]    | 0.8928 [NR]   |
