from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from torch.utils.data import DataLoader, Dataset
from os import listdir
from os.path import join
from PIL import Image

class Helper:
    @staticmethod
    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

    @staticmethod
    def calculate_valid_crop_size(crop_size, upscale_factor):
        return crop_size - (crop_size % upscale_factor)

    @staticmethod
    def train_hr_transform(crop_size):
        return Compose([
            RandomCrop(crop_size),
            ToTensor(),
        ])

    @staticmethod
    def train_lr_transform(crop_size, upscale_factor):
        return Compose([
            ToPILImage(),
            Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
            ToTensor()
        ])

    @staticmethod
    def display_transform():
        return Compose([
            ToPILImage(),
            Resize(400),
            CenterCrop(400),
            ToTensor()
        ])

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if Helper.is_image_file(x)]
        crop_size = Helper.calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = Helper.train_hr_transform(crop_size)
        self.lr_transform = Helper.train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)