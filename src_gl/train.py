from __future__ import print_function
import os
from math import log10

from datetime import datetime
from tqdm import tqdm
import logging
import logging.handlers

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from net_gl import _netlocalD,_netG

from TensorData_Loader import get_data_set
from tensorboardX import SummaryWriter

import argparse


# Training settings
parser = argparse.ArgumentParser(description='Globally and Locally Consistent Image Completion')
parser.add_argument('--label', default='compare_gl' , help='facades')
parser.add_argument('--datasetPath', default='../dataset/Facade' , help='facades')

parser.add_argument('--resume_epoch', default=0 , help='facades')
parser.add_argument('--nEpochs', type=int, default=3*1000, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=8*8, help='training batch size')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--testBatchSize', type=int, default=8*4, help='testing batch size')

parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--nef', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')

parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gan_wei', type=int, default=0.0004, help='weight on L1 term in objective')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

parser.add_argument('--G_model', type=str, default="gl")
parser.add_argument('--D_model', type=str, default="gl_d")

#-----Sett logging--------
logger = logging.getLogger('mylogger')
fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

streamHandler = logging.StreamHandler()
streamHandler.setFormatter(fomatter)
logger.addHandler(streamHandler)
logger.setLevel(logging.INFO)

def train(epoch):
    real_label = 1
    fake_label = 0

    for iteration, batch in tqdm(enumerate(training_data_loader, 1), total=len(training_data_loader)):
        input, target, input_masked = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        mask_tensor = input_masked[:, -1].unsqueeze(1)

        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        netD.zero_grad()
        real_cpu = target.to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device).unsqueeze(-1)

        output = netD(real_cpu, mask_tensor)
        D_real_loss = criterionGAN(output, label)
        D_real_loss.backward()
        D_x = output.mean().item()

        # train with fake
        fake = netG(input_masked)
        label.fill_(fake_label)
        output = netD(fake.detach(), mask_tensor)
        D_fake_loss = criterionGAN(output, label)
        D_fake_loss.backward()
        D_G_z1 = output.mean().item()
        loss_d = (D_real_loss + D_fake_loss) * opt.gan_wei
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)
        pred_fake = netD(fake, mask_tensor)

        loss_g_gan = criterionGAN(pred_fake, label)
        loss_g_l2 = criterionMSE(fake, target)
        loss_g = opt.gan_wei * loss_g_gan + loss_g_l2

        loss_g.backward()
        optimizerG.step()

    writer.add_scalar('dataD/d_loss', loss_d.item(), epoch)
    writer.add_scalar('dataD/D_real_loss', D_real_loss.item(), epoch)
    writer.add_scalar('dataD/D_fake_loss', D_fake_loss.item(), epoch)

    writer.add_scalar('dataG/loss_g', loss_g.item(), epoch)
    writer.add_scalar('dataG/loss_g_gan', loss_g_gan.item(), epoch)
    writer.add_scalar('dataG/loss_g_l2', loss_g_l2.item(), epoch)

def ToPilImage(image_tensor):
    image_norm = (image_tensor.data + 1 )/2
    return  image_norm


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def test(epoch):
    sum_psnr = 0
    prediction = 0
    for batch in testing_data_loader:
        input, target, input_masked = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        prediction = netG(input_masked)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        sum_psnr += psnr

    avg_psnr = sum_psnr / len(testing_data_loader)
    logger.info("[{}]===> Avg. PSNR: {:.4f} dB".format(epoch, avg_psnr))
    writer.add_image('Test/input', ToPilImage(input), epoch)
    writer.add_image('Test/prediction', ToPilImage(prediction), epoch)
    writer.add_image('Test/target', ToPilImage(target), epoch)
    writer.add_scalar('PSNR/PSNR', avg_psnr, epoch)

def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint/{}_{}".format(opt.label,opt.G_model))) :
        os.mkdir(os.path.join("checkpoint/{}_{}".format(opt.label,opt.G_model)))
    net_g_model_out_path = "checkpoint/{}_{}/netG_model_epoch_{}.pth".format(opt.label,opt.G_model, epoch)
    net_d_model_out_path = "checkpoint/{}_{}/netD_model_epoch_{}.pth".format(opt.label,opt.G_model, epoch)
    torch.save(netG, net_g_model_out_path)
    torch.save(netD, net_d_model_out_path)

    logger.info("Checkpoint saved to {}".format("checkpoint" + opt.label))

if __name__ ==  '__main__':
    opt = parser.parse_args()

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(opt.seed)
    device = torch.device("cuda" if opt.cuda else "cpu")

    logger.info('===> Loading datasets')
    train_set = get_data_set(opt.datasetPath, "train")
    test_set = get_data_set(opt.datasetPath, "test")
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                     shuffle=True)


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    logger.info('===> Building model')
    resume_epoch = 0
    if opt.resume_epoch > 0:
        resume_epoch = opt.resume_epoch
        net_g_model_out_path = "checkpoint/{}_{}/netG_model_epoch_{}.pth".format(opt.label, opt.G_model, resume_epoch)
        net_d_model_out_path = "checkpoint/{}_{}/netD_model_epoch_{}.pth".format(opt.label, opt.G_model, resume_epoch)
        netG = torch.load(net_g_model_out_path)
        netD = torch.load(net_d_model_out_path)
    else:
        netG = _netG(opt)
        netG.apply(weights_init)
        netD = _netlocalD(opt.nc, opt.ndf, n_layers=3)
        netD.apply(weights_init)

    criterionGAN = nn.BCELoss()
    criterionMSE = nn.MSELoss()

    # setup optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    logger.info('---------- Networks initialized -------------')
    print_network(netG)
    print_network(netD)
    logger.info('-----------------------------------------------')

    netD = netD.to(device)
    netG = netG.to(device)
    criterionGAN = criterionGAN.to(device)
    criterionMSE = criterionMSE.to(device)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs_'+opt.label, opt.label + "_" + str(opt.resume_epoch) + "_" + current_time)
    writer = SummaryWriter(log_dir)

    for epoch in range(resume_epoch, opt.nEpochs + 1):
        train(epoch)
        test(epoch)
        if epoch % 10 == 0:        checkpoint(epoch)

    writer.close()


