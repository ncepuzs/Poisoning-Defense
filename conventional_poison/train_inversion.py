from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os, shutil
from data import FaceScrub, CelebA, BinaryDataset, extract_dataset, Partial_Dataset, Poisoned_Dataset
from model import Classifier, Inversion
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.datasets as dsets
import logging
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from poisoning import generate_poison


import math
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument('--batch-size', type=int, default=128, metavar='')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='')
parser.add_argument('--epochs', type=int, default=100, metavar='')
parser.add_argument('--lr', type=float, default=0.01, metavar='')
parser.add_argument('--momentum', type=float, default=0.5, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=10, metavar='')
# parser.add_argument('--nc', type=int, default=1)
# parser.add_argument('--ndf', type=int, default=128)
# parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--nz', type=int, default=10)
parser.add_argument('--truncation', type=int, default=10)
parser.add_argument('--c', type=float, default=50.)
parser.add_argument('--num_workers', type=int, default=0, metavar='')
parser.add_argument('--path_out', type=str, default='vector-based/')
parser.add_argument('--early_stop', type=int, default=15)
parser.add_argument('--group_size', type=int, default=100)


def train(inversion, log_interval, device, data_loader, optimizer, epoch, logger):
    inversion.train()
    print(len(data_loader))
    for batch_idx, (data, target) in enumerate(data_loader):
        data, prediction = data.to(device), target.to(device)
        optimizer.zero_grad()

        reconstruction = inversion(prediction)
        loss = F.mse_loss(reconstruction, data)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                 len(data_loader.dataset), loss.item()))
            # logging : comparison
            # logger.info("----------------------------------------------------")
            # logger.info("prediction_original:\t{}".format(prediction[0:1]))
            # logger.info("prediction_poisoning:\t{}".format(prediction_p))
            # logger.info("----------------------------------------------------")


def test(classifier, inversion, device, data_loader, epoch, msg, logger, path_out):
    classifier.eval()
    inversion.eval()
    mse_loss = 0
    plot = True
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            prediction = classifier(data, release=True)
            reconstruction = inversion(prediction)
            mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()

            if plot:
                truth = data[0:512]
                inverse = reconstruction[0:512]
                out = torch.cat((inverse, truth))
                for i in range(16):
                    out[i * 64:i * 64 + 32] = inverse[i * 32:i * 32 + 32]
                    out[i * 64 + 32:i * 64 + 64] = truth[i * 32:i * 32 + 32]
                vutils.save_image(out, path_out + 'recon_{}.png'.format(epoch), nrow=32, normalize=False)
                plot = False

    mse_loss /= len(data_loader.dataset) * 32 * 32
    logger.info('\nTest inversion model on {} set: Average MSE loss: {:.6f}\n'.format(msg, mse_loss))
    return mse_loss


def main():
    torch.autograd.set_detect_anomaly(True)
    args = parser.parse_args()

    os.makedirs(args.path_out, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        filename=args.path_out + 'loss.log',
                        filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认果不写的话，就是追加模式
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d]: %(message)s'
                        # 日志格式
                        )
    logger = logging.getLogger(__name__)

    logger.info("================================")
    logger.info(args)
    logger.info("================================")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda == False:
        logger.info('GPU is not used')
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if use_cuda else {}

    torch.manual_seed(args.seed)

    # transform = transforms.Compose([transforms.ToTensor()])

    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    # transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])
                                    ])

    train_dataset = dsets.MNIST(root='./data/',
                                train=True,
                                transform=transform,
                                download=True)
    print("len of train_dataset:", len(train_dataset))
    train_dataset = Partial_Dataset(train_dataset, 1000)
    print("len of partial train_dataset:", len(train_dataset))

    test_dataset = dsets.MNIST(root='./data/',
                               train=False,
                               transform=transform)
    print("len of test_dataset:", len(test_dataset))
    test_dataset = Partial_Dataset(test_dataset, 1000)
    print("len of partial test_dataset:", len(test_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader_err = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    shadow_loader_list = extract_dataset(args.nz, train_dataset, 6000, 5000, 100)
    # tst_dataloader_list = extract_dataset(args.nz, test_set, 50, 50, 1)

    classifier = nn.DataParallel(Classifier(nz=args.nz)).to(device)
    # Load classifier
    path = args.path_out + 'classifier.pth'
    checkpoint = torch.load(path)

    classifier.load_state_dict(checkpoint['model'])
    # print("test success")
    epoch = checkpoint['epoch']
    best_cl_acc = checkpoint['best_cl_acc']
    print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_cl_acc))

    classifier.eval()

    inversion = nn.DataParallel(Inversion(nz=args.nz, truncation=args.truncation, c=args.c)).to(device)
    optimizer = optim.Adam(inversion.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)

    # load substitute inversion
    inversion_sub = nn.DataParallel(Inversion(nz=args.nz, truncation=args.truncation, c=args.c)).to(device)
    # param_path = 'vector_based/poisoning/inversion_sub.pth'
    # checkpoint = torch.load(param_path)
    # inversion_sub.load_state_dict(checkpoint['model'])

    # generate poisoning examples: return poisoned data_loader
    classifier.eval()
    group_size = args.group_size
    checkpoint_s = torch.load('vector_based/poisoning/inversion_sub.pth')
    # print(checkpoint_s)
    print('type of checkpoint_s:', type(checkpoint_s))
    poison_gen_loader = torch.utils.data.DataLoader(train_dataset, batch_size=group_size, shuffle=True, **kwargs)
    train_dataset_poi = Poisoned_Dataset(poison_gen_loader, generate_poison, classifier, inversion, checkpoint_s, device, group_size)
    poison_loader = torch.utils.data.DataLoader(train_dataset_poi, batch_size=group_size, shuffle=True, **kwargs)

    # Train inversion model
    best_recon_loss = 999
    early_stop_label = 0
    for epoch in range(1, args.epochs + 1):
        train(inversion, args.log_interval, device, poison_loader, optimizer, epoch, logger)
        recon_loss = test(classifier, inversion, device, test_loader, epoch, 'test1', logger, args.path_out)
        # test(classifier, inversion, device, test2_loader, epoch, 'test2')

        if recon_loss < best_recon_loss:
            best_recon_loss = recon_loss
            state = {
                'epoch': epoch,
                'model': inversion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_recon_loss': best_recon_loss
            }
            torch.save(state, args.path_out + 'inversion.pth')
            shutil.copyfile(args.path_out + 'recon_{}.png'.format(epoch), args.path_out + 'best.png')
            # shutil.copyfile('out/recon_test2_{}.png'.format(epoch), 'out/best_test2.png')

            early_stop_label = 0
        else:
            early_stop_label += 1
            if early_stop_label == args.early_stop:
                break


if __name__ == '__main__':
    main()
