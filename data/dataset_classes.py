import os
import data.utils as utils
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image

class TrainDatasetFromFolder(Dataset):
    def __init__(self, image_list, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()

        self.image_filenames = image_list
        crop_size = utils.calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = utils.train_hr_transform(crop_size)
        self.lr_transform = utils.train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):        
        img = Image.open(self.image_filenames[index]).convert('RGB')
        hr_image = self.hr_transform(img)
        lr_image = self.lr_transform(hr_image)
        hr_image = (hr_image / 0.5) - 1
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class ValDatasetFromFolder(Dataset):

    def __init__(self, image_list, crop_size, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()

        self.image_filenames = image_list
        crop_size = utils.calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = utils.train_hr_transform(crop_size)
        self.lr_transform = utils.train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        img = Image.open(self.image_filenames[index]).convert('RGB')
        hr_image = self.hr_transform(img)
        lr_image = self.lr_transform(hr_image)
        hr_image = (hr_image / 0.5) - 1
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class TestDatasetFromFolder(Dataset):
    def __init__(self, image_list, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        
        self.image_filenames = image_list
        self.upscale_factor = upscale_factor

    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        img = Image.open(img_name).convert('RGB')
        hr_image = ToTensor()(img)
        if hr_image.size(1) % self.upscale_factor != 0 and hr_image.size(2) % self.upscale_factor != 0:
            height_diff = hr_image.size(1) % self.upscale_factor
            width_diff = hr_image.size(2) % self.upscale_factor
            hr_image = utils.test_resize(hr_image.size(1) - height_diff, hr_image.size(2) - width_diff)(hr_image)

        elif hr_image.size(1) % self.upscale_factor != 0 and hr_image.size(2) % self.upscale_factor == 0:
            height_diff = hr_image.size(1) % self.upscale_factor
            hr_image = utils.test_resize(hr_image.size(1) - height_diff, hr_image.size(2))(hr_image)

        elif hr_image.size(1) % self.upscale_factor == 0 and hr_image.size(2) % self.upscale_factor != 0:
            width_diff = hr_image.size(2) % self.upscale_factor
            hr_image = utils.test_resize(hr_image.size(1), hr_image.size(2) - width_diff)(hr_image)

        lr_transform = utils.test_lr_transform(
            height=hr_image.size(1),
            width=hr_image.size(2),
            upscale_factor=self.upscale_factor
        )
        lr_image = lr_transform(hr_image)
        hr_image = (hr_image / 0.5) - 1
        return lr_image, hr_image, img_name

    def __len__(self):
        return len(self.image_filenames)