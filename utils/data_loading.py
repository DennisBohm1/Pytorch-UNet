import glob
import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from monai.data import DataLoader, Dataset, ITKReader, ImageDataset, CacheDataset
from monai.transforms import *
from torch.utils.data import random_split


def create_data_loader(dir_img, dir_mask, val_percent, batch_size):
    reader = ITKReader()
    combined_list = []
    for img in listdir(Path(dir_img)):
        if img.endswith('.nii.gz'):
            combined_list.append({
                'image': str(Path(dir_img)) + '/' + str(img),
                'mask': str(Path(dir_mask)) + '/label' + str(img),
            })



    test_transforms = Compose(
        [
            LoadImaged(keys="image"),  # d, h, w
            # AddChanneld(keys="image"),  # c, d, h, w
            # Transposed(keys="image", indices=[0, 2, 3, 1]),  # c, w, h, d
            # Lambdad(keys="image", func=lambda x: x / x.max()),
            #         SpatialPadd(keys="image", spatial_size=cfg.img_size),  # in case less than 80 slices
            EnsureTyped(keys="image", dtype=torch.float32),
            LoadImaged(keys="mask"),  # d, h, w
            # AddChanneld(keys="mask"),  # c, d, h, w
            # Transposed(keys="mask", indices=[0, 2, 3, 1]),  # c, w, h, d
            # Lambdad(keys="image", func=lambda x: x / x.max()),
            #         SpatialPadd(keys="image", spatial_size=cfg.img_size),  # in case less than 80 slices
            EnsureTyped(keys="mask", dtype=torch.float32),
        ]
    )

    # dataset = ImageDataset(
    #     image_files=glob.glob('../data/NIH/Images/*.nii.gz'),
    #     seg_files=glob.glob('../data/NIH/Mask/*.nii.gz'),
    #     transform=test_transforms,
    #     # seg_transform=test_transforms
    # )

    dataset = Dataset(combined_list, transform=test_transforms)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    # 3. Create data loaders
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=batch_size, num_workers=0)

    # for batch in train_loader:
    #     images = batch[0]
    #     true_masks = batch[1]

    return train_loader, val_loader


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        # if not is_mask:
        #     if img_ndarray.ndim == 2:
        #         img_ndarray = img_ndarray[np.newaxis, ...]
        #     else:
        #         img_ndarray = img_ndarray.transpose((2, 0, 1))
        #
        #     img_ndarray = img_ndarray / 255

        return img_ndarray

    def load_images(fs, N, Nchannels, ImSize):
        if N < 0:
            N = fs.__len__()

        X = np.ndarray(shape=(N, Nchannels, ImSize, ImSize), dtype=np.uint8)
        Y = np.ndarray(shape=(N, 1), dtype=np.uint8)
        for i in range(0, N):
            image = Image.open(fs[i], 'r')
            X[i, :, :, :] = np.swapaxes(image, 0, 2)

            # print extension
            Y[i] = 1

            # progress
            if i % 10000 is 0:
                print
                "{0} of {1}: {2}".format(i, N, fs[i])

        return (X, Y, fs[0:N])

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load_images(mask_file[0], 3, 1, )
        img = self.load_images(img_file[0])

        # assert img.size == mask.size, \
        #     f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # mask = self.preprocess(mask_file, self.scale, is_mask=True)
        # img = self.preprocess(img_file, self.scale, is_mask=False)

        return {
            'image': torch.as_tensor(img_file.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask_file.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
