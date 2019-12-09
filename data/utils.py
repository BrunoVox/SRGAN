import torch
import torchvision.transforms as t
from PIL import Image
import os
import random
import pickle

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def normalize_tensor(tensor):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
    tensor = (tensor - mean) / std
    return tensor

def train_lr_transform(crop_size, upscale_factor):
    return t.Compose(
        [
            t.ToPILImage(),
            t.Resize((crop_size // upscale_factor, crop_size // upscale_factor), interpolation=Image.BICUBIC),
            t.ToTensor()
        ]
    )

def train_hr_transform(crop_size):
    return t.Compose(
        [
            # ToPILImage(),
            # Resize(crop_size, interpolation=Image.BICUBIC),
            t.RandomCrop(crop_size),
            t.ToTensor()
        ]
    )

def test_lr_transform(height, width, upscale_factor):
    return t.Compose(
        [
            t.ToPILImage(),
            t.Resize((height // upscale_factor, width // upscale_factor), interpolation=Image.BICUBIC),
            t.ToTensor()
        ]
    )

def test_resize(height, width):
    return t.Compose(
        [
            t.ToPILImage(),
            t.Resize((height, width), interpolation=Image.BICUBIC),
            t.ToTensor()
        ]
    )

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_partitions(img_dir):
    train_file = 'data_state/train.pkl'
    val_file = 'data_state/val.pkl'
    cwd = os.getcwd()
    file_path = f'{cwd}/data/data_state'
    if os.path.isfile(train_file) and os.path.isfile(val_file):
        with open(train_file, 'rb') as f:
            img_train = pickle.load(f)
        with open(val_file, 'rb') as f:
            img_val = pickle.load(f)
    else:
        create_path(file_path)
        img_paths = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if is_image_file(x)]
        img_train = [img_paths[i] for i in random.sample(range(len(img_paths)), int(len(img_paths) * 0.85))]
        img_train_set = set(img_train)
        img_paths_set = set(img_paths)
        img_val_set = img_paths_set - img_train_set
        img_val = list(img_val_set)
        with open(f'{file_path}/train.pkl', 'wb') as f:
            pickle.dump(img_train, f)
        with open(f'{file_path}/val.pkl', 'wb') as f:
            pickle.dump(img_val, f)
    return img_train, img_val