from data.dataset_classes import TrainDatasetFromFolder, ValDatasetFromFolder, TestDatasetFromFolder
from torch.utils.data import DataLoader

def TrainDataloader(image_list, crop_size, upscale_factor):
    dataset = TrainDatasetFromFolder(
        image_list, 
        crop_size=crop_size, 
        upscale_factor=upscale_factor
    )
    datasetLoader = DataLoader(
        dataset=dataset,
        num_workers=8,
        batch_size=16,
        shuffle=True,
        pin_memory=True
    )
    return datasetLoader

def ValDataloader(image_list, crop_size, upscale_factor):
    dataset = ValDatasetFromFolder(
        image_list, 
        crop_size=crop_size, 
        upscale_factor=upscale_factor
    )
    datasetLoader = DataLoader(
        dataset=dataset,
        num_workers=8,
        batch_size=16,
        shuffle=True,
        pin_memory=True
    )
    return datasetLoader

def TestDataloader(image_list, upscale_factor):
    dataset = TestDatasetFromFolder(
        image_list, 
        upscale_factor=upscale_factor
    )
    datasetLoader = DataLoader(
        dataset=dataset,
        num_workers=8,
        batch_size=1,
        shuffle=False
    )
    return datasetLoader