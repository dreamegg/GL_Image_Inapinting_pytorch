from __future__ import print_function
import argparse
import os

import torch
from ssim_torch import ssim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import math

from TensorData_Loader import save_img, get_data_set

# Testing settings
parser = argparse.ArgumentParser(description='2 Stage RestoNet-PyTorch-implementation')
parser.add_argument('--label', default='compare_gl' , help='facades')
parser.add_argument('--datasetPath', default='../dataset/Facade' , help='facades')

parser.add_argument('--G_model', type=str, default="gl")

parser.add_argument('--resume_epoch', default=90 , help='facades')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')

parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')

opt = parser.parse_args()
print(opt)
device = torch.device("cuda" if opt.cuda else "cpu")

if not os.path.exists("result"):
    os.mkdir("result")
if not os.path.exists(os.path.join("result/{}".format(opt.label))):
    os.mkdir(os.path.join("result/{}".format(opt.label)))

f= open("result/{}.txt".format(opt.label),'w')
avg_psnr_2 = 0
sum_psnr_2 = 0
avg_ssim_2 = 0
sum_ssim_2 = 0
avg_l1 = 0
sum_l1 = 0
avg_l2 = 0
sum_l2 = 0
prediction = 0
count = 0

test_set = get_data_set(opt.datasetPath, "test")
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)
netG = torch.load("checkpoint/{}_{}/netG_model_epoch_{}.pth".format(opt.label, opt.G_model, opt.resume_epoch))
netG.to(device)
i =0

criterionMSE = nn.MSELoss()
criterionMSE = criterionMSE.cuda()

for batch in testing_data_loader:
    count = count+1

    input, target, input_masked = batch[0].to(device), batch[1].to(device), batch[2].to(device)
    mask_tensor = input_masked[:, -1].unsqueeze(1)

    netG.zero_grad()
    out = netG(input_masked)
    out = out.detach()

    loc = np.argwhere(np.asarray(1 - mask_tensor) > 0.1)
    (_, _, ystart, xstart) = loc.min(0)
    (_, _, ystop, xstop) = loc.max(0) + 1

    target_crop = target[:, :,xstart:xstop, ystart:ystop]
    out_crop = out[:, :,xstart:xstop, ystart:ystop]

    input_img = input.cpu().data[0]
    out_img = out.cpu().data[0]
    target_img = target.cpu().data[0]
    merged_result = torch.cat((input_img, out_img, target_img), 2)
    save_img(merged_result, "result/{}/{}_{}.jpg".format(opt.label, count, opt.label))

    p = 0
    l1 = 0
    l2 = 0
    fake = out_crop.cpu().data.numpy()
    real_center = target_crop.cpu().data.numpy()

    t = real_center - fake
    l2 = np.mean(np.square(t))
    l1 = np.mean(np.abs(t))
    real_center = (real_center + 1) * 127.5
    fake = (fake + 1) * 127.5

    def psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


    for i in range(1):
        p = p + psnr(real_center[i].transpose(1, 2, 0), fake[i].transpose(1, 2, 0))


    ssim_none = ssim(out_crop, target_crop)
    sum_l1 = sum_l1 + l1
    sum_l2 = sum_l2 + l2
    sum_psnr_2 = sum_psnr_2 + p
    sum_ssim_2 = sum_ssim_2 + ssim_none

    outstr = "[Global-local] [%4d]-> " % count
    outstr = outstr + "l1:\t%f\t" % l1
    outstr = outstr + "l2:\t%f\t" % l2
    outstr = outstr + "psnr:\t%f\t" % p
    outstr = outstr + "ssim_img :\t%f\t \n" % ssim_none
    print(outstr)
    f.write(outstr)



outresultstr = "[Global-local] Total-> psnr:\t%f\t" % (sum_psnr_2/ count)
outresultstr = outresultstr +  "ssim_img :\t%f\t \n" % (sum_ssim_2 / count)
outresultstr = outresultstr +  "l1 :\t%f\t \n" % (sum_l1 *100 / count)
outresultstr = outresultstr +  "l2 :\t%f\t \n" % (sum_l2 *100 / count)
print(outresultstr)
f.write(outresultstr)
f.close()