from PIL import Image
import numpy as np
import torch
import os.path
import torchvision.transforms as transforms
import random
import torch.utils.data as data


def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader


class CustomDatasetDataLoader():
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


def CreateDataset(opt):
    dataset = AlignedDataset()
    dataset.initialize(opt)
    return dataset


class AlignedDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        dir_A = '_A'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        dir_B = '_B'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path)

        params = get_params()
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A.convert('RGB'))

        B_path = self.B_paths[index]
        B = Image.open(B_path).convert('RGB')
        B_tensor = transform_A(B)

        input_dict = {'image1': A_tensor, 'image2': B_tensor, 'path1': A_path, 'path2': B_path}
        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def CameraPoseRead(self, dir):
        camera_pose_path = dir
        camera_pose = []

        f = open(camera_pose_path)
        for i in range(4):
            line = f.readline()
            tmp = line.split()
            camera_pose.append(tmp)
        camera_pose = np.array(camera_pose, dtype=np.float32)

        return camera_pose


def get_params():
    flip = random.random() > 0.5
    return {'flip': flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if opt.resize_or_crop == 'none':
        base = float(2 ** 3)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.txt', '.npz'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
