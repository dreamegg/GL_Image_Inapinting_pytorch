from os.path import join
from os import listdir

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

def load_tensor(filepath):
    img_tensor = torch.load(filepath)
    return img_tensor


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

def save_out(source_image_tensor, tearget_image_tensor, filename):
    image_numpy = tearget_image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    image_numpy.resize()
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()

        self.image_path = join(image_dir)
        self.subdir = "/tensor_tranced"
        self.image_filenames = [x for x in listdir(self.image_path+self.subdir )]

    def __getitem__(self, index):
        # Load Image
        loaded_tensor = load_tensor(join(self.image_path+self.subdir , self.image_filenames[index]))
        input = loaded_tensor[:3]
        input_masked = loaded_tensor[:4]
        target = loaded_tensor[4:7]
        tensor_size = loaded_tensor.size()

        return input, target, input_masked

    def __len__(self):
        return len(self.image_filenames)


def get_data_set(root_dir, set_type):
    if set_type == "train" :
        dataset_dir = join(root_dir, "train")
    elif set_type == "test" :
        dataset_dir = join(root_dir, "test")
    return DatasetFromFolder(dataset_dir)
