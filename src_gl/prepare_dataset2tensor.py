from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random

import torch.utils.data as data

from os.path import exists, join, basename
from os import makedirs, remove, listdir
from tqdm import tqdm

from PIL import Image
import torchvision.transforms as transforms

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Image devider for trainset')
parser.add_argument('--in_dir', default="../dataset/facade/test", help="super resolution upscale factor")
parser.add_argument('--out_dir', default="../dataset/facade/test", help="super resolution upscale factor")
parser.add_argument('--ImgFormat', default="RGB", help="RGV/YUV")

parser.add_argument('--patchSize', type=int, default=128)
parser.add_argument('--masked_size', type=int, default=64)
parser.add_argument('--boundary_margin', type=int, default=16)

parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=8, help='number of epochs to train for')
parser.add_argument('--nRepeat', type=int, default=4, help='number of epochs to train for')
parser.add_argument('--cuda', default='True', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()
print(opt)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath, format):
    img = Image.open(filepath).convert(format)
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, opt):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(opt.in_dir, x) for x in listdir(opt.in_dir) if is_image_file(x)]
        self.targetdir = opt.out_dir
        self.nRepeat = opt.nRepeat
        self.ImageFormat = opt.ImgFormat

        self.patchSize = opt.patchSize
        self.masked_size = opt.masked_size
        self.boundary_margin = opt.boundary_margin

        self.reszie = transforms.Compose([transforms.Resize((356, 436)), transforms.RandomCrop((opt.patchSize, opt.patchSize))])
        self.crop = transforms.Compose([transforms.RandomCrop((opt.patchSize, opt.patchSize))])

        self.transform = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.Totensor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index], self.ImageFormat)
        width, height = input.size

        if (width < height) :
            img_ratio = height / width
            Reszied_size = random.randrange(self.patchSize, width)
            input = input.resize((Reszied_size, int(Reszied_size*img_ratio)))

        else :
            img_ratio = width /height
            Reszied_size = random.randrange(self.patchSize,height)
            input = input.resize((int(Reszied_size*img_ratio), Reszied_size))

        save_string = self.image_filenames[index].split('\\')[-1]
        save_string = save_string.split('.')[0]
        for i in range(self.nRepeat) :
            startx = random.randrange(self.boundary_margin,(self.patchSize - self.masked_size)-self.boundary_margin)
            starty = random.randrange(self.boundary_margin,(self.patchSize - self.masked_size)-self.boundary_margin)

            trans_img = self.crop(input)
            out_tensor = self.transform(trans_img)

            blank_img = Image.new("L", (self.masked_size, self.masked_size))
            trans_img.paste(blank_img, (startx, starty))

            mask_img = Image.new("L", (self.patchSize, self.patchSize), 255)
            mask_img.paste(blank_img, (startx, starty))

            input_tensor = self.transform(trans_img)
            mask = self.Totensor(mask_img)
            input_masked = torch.cat((input_tensor, mask), 0)
            result = torch.cat((input_masked, out_tensor), 0)

            torch.save(result, self.targetdir + "\\tensor_tranced\\" + save_string + "_{}_{}.ts".format(epoch,i))

        return 0

    def __len__(self):
        return len(self.image_filenames)

torch.manual_seed(opt.seed)
device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')
root_dir = opt.in_dir
if not exists(opt.out_dir+"\\tensor_tranced"):    makedirs(opt.out_dir+"\\tensor_tranced")

train_set =  DatasetFromFolder(opt)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

def devide(epoch):
    count =0
    for iteration in tqdm(enumerate(training_data_loader, 1), total=len(training_data_loader) ):
        count = 0

for epoch in range(1, opt.nEpochs + 1):
    devide(epoch)