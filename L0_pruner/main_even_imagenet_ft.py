from __future__ import print_function
import os
import datetime
import csv
import argparse
import shutil
import numpy as np
import time
import warnings
import distutils
import distutils.util
from contextlib import redirect_stdout
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models as models
import numpy
import math
import random
from importlib import import_module
# Training settings
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--dataset', type=str, default='imagenet',
                    help='training dataset (default: cifar10)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus',type=int,nargs='+',default=[0,1],help='Select gpu_id to use. default:[1]',)
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('-opt', '--optimizer', metavar='OPT', default="SGD",
                    help='optimizer algorithm')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=90, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default=''
                    , type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--arch', default='resnet', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=50, type=int,
                    help='depth of the neural network')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-model', default=True, type=lambda x:bool(distutils.util.strtobool(x)),
                    help='For Saving the current Model (default: True)')
def main():
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
    criterion = torch.nn.CrossEntropyLoss().to(device)

    print('loading data...')
    train_dataset = datasets.ImageFolder('./imagenet/train',
                                         transforms.Compose([transforms.RandomResizedCrop(224),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                                  std = [0.229, 0.224, 0.225]),
                                                             ])
                                         )
    val_dataset = datasets.ImageFolder('./imagenet/val',
                                       transforms.Compose([transforms.Resize(256),
                                                           transforms.CenterCrop(224),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                                std = [0.229, 0.224, 0.225])
                                                           ]))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               shuffle = True,
                                               num_workers = args.workers,
                                               pin_memory = True)
    test_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size = args.batch_size,
                                              shuffle = False,
                                              num_workers = args.workers,
                                              pin_memory = True)

    if args.refine:
        checkpoint = torch.load(args.refine)
        layer_cfg = checkpoint['cfg']
        model = import_module(f'models.{args.arch}').resnet50_flops(layer_cfg = layer_cfg).to(device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> creating model '{}'".format(args.arch))
        model = import_module(f'models.{args.arch}').resnet50_flops().to(device)


    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum = args.momentum, weight_decay = args.weight_decay)

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids = args.gpus)

    current_name = ""
    model_name = '%s%s' % (args.arch,current_name)

    if (args.save_model):
        model_dir = os.path.join(os.path.join(os.path.join(os.getcwd(), "../models"), args.dataset), model_name)
        if not os.path.isdir(model_dir):
             os.makedirs(model_dir, exist_ok=True)

    def adjust_learning_rate(optimizer, epoch, step, len_epoch):
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))

        """Warmup"""
        if epoch < 5:
            lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(train_loader, model, criterion, optimizer, epoch, args):
        batch_time = AverageMeter('Time', ':.4f')
        data_time = AverageMeter('Data', ':.4f')
        losses = AverageMeter('Loss', ':.4f')
        top1 = AverageMeter('Acc@1', ':.2f')
        top5 = AverageMeter('Acc@5', ':.2f')
        progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                                 prefix = "Epoch: [{}]".format(epoch))
        model.train()
        end = time.time()
        for batch_idx, data in enumerate(train_loader):
            data_time.update(time.time() - end)
            input = data[0].to(device)
            target = data[1].to(device)
            train_loader_len = int(math.ceil(len(train_loader) / args.batch_size))

            adjust_learning_rate(optimizer, epoch, batch_idx, train_loader_len)

            output = model(input)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk = (1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                progress.print(batch_idx)

        return (losses.avg, top1.avg.cpu().numpy(), top5.avg.cpu().numpy(), batch_time.avg)

    def test(val_loader, model, criterion, args):
        batch_time = AverageMeter('Time', ':.4f')
        losses = AverageMeter('Loss', ':.4f')
        top1 = AverageMeter('Acc@1', ':.2f')
        top5 = AverageMeter('Acc@5', ':.2f')
        progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5, prefix = 'Test: ')
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, data in enumerate(val_loader):
                input = data[0].to(device)
                target = data[1].to(device)
                output = model(input)
                loss = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk = (1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.print(i)
            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f}  * Acc@5 {top5.avg:.3f}'.format(top1 = top1, top5 = top5))

        return (losses.avg, top1.avg.cpu().numpy(), top5.avg.cpu().numpy(), batch_time.avg)

    def save_checkpoint(state, is_best, filepath):
        torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

    class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self, name, fmt=':f'):
            self.name = name
            self.fmt = fmt
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

        def __str__(self):
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)

    class ProgressMeter(object):
        def __init__(self, num_batches, *meters, prefix=""):
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
            self.meters = meters
            self.prefix = prefix

        def print(self, batch):
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
            print('\t'.join(entries))

        def _get_batch_fmtstr(self, num_batches):
            num_digits = len(str(num_batches // 1))
            fmt = '{:' + str(num_digits) + 'd}'
            return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    best_prec1 = 0.
    best_prec5 = 0.

    for epoch in range(args.start_epoch, args.epochs):
        train_epoch_log = train(train_loader, model, criterion, optimizer, epoch, args)
        test_epoch_log = test(test_loader, model, criterion, args)

        acc1 = test_epoch_log[1]
        acc5 = test_epoch_log[2]

        is_best = acc1 > best_prec1

        best_prec1 = max(acc1, best_prec1)
        best_prec5 = max(acc5, best_prec5)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict() if len(args.gpus) > 1 else model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            'optimizer': optimizer.state_dict(),
            }, is_best, filepath = model_dir)
    print("Best accuracy1: " + str(best_prec1))
    print("Best accuracy5: " + str(best_prec5))
    with open(os.path.join(model_dir, 'BestAcc_' + '.txt'), 'w') as weights_log_file:
        with redirect_stdout(weights_log_file):
            print("Best Top1 accuracy: " + str(best_prec1))
            print("Best Top5 accuracy: " + str(best_prec5))
if __name__ == '__main__':
    main()
