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
from importlib import import_module
# Training settings
parser = argparse.ArgumentParser(description='PyTorch training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--percent', type=float, default=0.7,
                    help='global pruning rate (default: 0.7)')
parser.add_argument('-opt', '--optimizer', metavar='OPT', default="SGD",
                    help='optimizer algorithm')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
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
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-model', default=True, type=lambda x:bool(distutils.util.strtobool(x)),
                    help='For Saving the current Model (default: True)')

def main():
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}

    # Dataset
    if args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data/cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data/cifar10', train=False, download=True,
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data/cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data/cifar100', train=False, download=True,transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # Create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'vgg':
        model = models.__dict__[args.arch](depth=args.depth).to(device)
    elif args.arch == 'resnet_cifar':
        if args.depth == '56':
            model = import_module(f'models.{args.arch}').resnet('resnet56').to(device)
        elif args.depth == '110':
            model = import_module(f'models.{args.arch}').resnet('resnet110').to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [80, 120], gamma = 0.1)

    current_name = ""
    model_name = '%s%s' % (args.arch,current_name)

    if (args.save_model):
        model_dir = os.path.join(os.path.join(os.path.join(os.getcwd(), "../models"), args.dataset), model_name)
        if not os.path.isdir(model_dir):
             os.makedirs(model_dir, exist_ok=True)

    def projection(x, size, size_x):
        y = torch.zeros(size).to(device)
        x_max = torch.abs(x).to(device)
        values, indices = torch.topk(x_max, k = size_x)
        for index, value in enumerate(indices.to(device)):
            y[value] = x[value]
        return y

    def train_L0(train_loader, model, criterion, optimizer, epoch, args):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,prefix="Epoch: [{}]".format(epoch))
        model.train()
        end = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.arch == 'vgg':
                gamma_del_all = []
                gamma_size = 0
                index = 0
                layer_numer = 0
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        y_hat = m.weight.data
                        size = m.weight.data.shape[0]
                        indice = torch.argmax(torch.abs(y_hat),dim = 0).to(device)
                        y_hat_del = torch.cat((y_hat[:indice], y_hat[(indice + 1):])).to(device)
                        gamma_del_all.append(y_hat_del.view(-1))
                        gamma_size = gamma_size + size
                        layer_numer = layer_numer + 1
                gamma_del_all = torch.cat(gamma_del_all).to(device)
                size_x = int((1 -args.percent) * gamma_size)-layer_numer
                gamma_all_aferpro = projection(gamma_del_all, len(gamma_del_all), size_x).to(device)

                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        yb_hat = m.weight.data
                        size = m.weight.data.shape[0]
                        indice = torch.argmax(torch.abs(yb_hat),dim = 0).to(device)
                        new_y_hat_del = gamma_all_aferpro[index:(index + size - 1)].to(device)
                        left = new_y_hat_del[:indice].to(device)
                        right = new_y_hat_del[indice:].to(device)
                        new_left = torch.cat([left,torch.tensor([yb_hat[indice]]).to(device)], dim=0).to(device)
                        new_y_hat = torch.cat([new_left,right], dim=0).to(device)
                        m.weight.data = new_y_hat
                        index = index + size - 1
            elif args.arch == 'resnet_cifar':
                gamma_del_all = []
                gamma_size = 0
                index = 0
                layer_numer = 0
                bn_first_layer = 0
                for name, module in model.named_modules():
                    if "bn1" in name:
                        if bn_first_layer == 0:
                            bn_first_layer = bn_first_layer + 1
                            continue
                        else:
                            y_hat = module.weight.data
                            size = module.weight.data.shape[0]
                            indice = torch.argmax(torch.abs(y_hat), dim = 0).to(device)
                            y_hat_del = torch.cat((y_hat[:indice], y_hat[(indice + 1):])).to(device)
                            gamma_del_all.append(y_hat_del.view(-1))
                            gamma_size = gamma_size + size
                            layer_numer = layer_numer + 1
                gamma_del_all = torch.cat(gamma_del_all).to(device)
                size_x = int((1 - args.percent) * gamma_size) - layer_numer
                gamma_all_aferpro = projection(gamma_del_all, len(gamma_del_all), size_x).to(device)
                bn_first_layer = 0

                for name, module in model.named_modules():
                    if "bn1" in name:
                        if bn_first_layer == 0:
                            bn_first_layer = bn_first_layer + 1
                            continue
                        else:
                            yb_hat = module.weight.data
                            size = module.weight.data.shape[0]
                            indice = torch.argmax(torch.abs(yb_hat), dim = 0).to(device)
                            new_y_hat_del = gamma_all_aferpro[index:(index + size - 1)].to(device)
                            left = new_y_hat_del[:indice].to(device)
                            right = new_y_hat_del[indice:].to(device)
                            new_left = torch.cat([left, torch.tensor([yb_hat[indice]]).to(device)], dim = 0).to(device)
                            new_y_hat = torch.cat([new_left, right], dim = 0).to(device)
                            module.weight.data = new_y_hat
                            index = index + size - 1

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                progress.print(batch_idx)

        return (losses.avg, top1.avg.cpu().numpy(), batch_time.avg)

    def train(train_loader, model, criterion, optimizer, epoch, args):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,prefix="Epoch: [{}]".format(epoch))
        model.train()
        end = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                progress.print(batch_idx)

        return (losses.avg, top1.avg.cpu().numpy(), batch_time.avg)

    def test(val_loader, model, criterion, args):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(val_loader), batch_time, losses, top1, prefix='Test: ')
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i,(data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), data.size(0))
                top1.update(acc1[0], data.size(0))
                top5.update(acc5[0], data.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.print(i)
        # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

        return (losses.avg, top1.avg.cpu().numpy(), batch_time.avg)

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


    start_time = time.time()
    best_prec1 = 0.
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in range(10):
            train_epoch_log = train(train_loader, model, criterion, optimizer, epoch, args)
        else:
            train_epoch_log = train_L0(train_loader, model, criterion, optimizer, epoch, args)
        scheduler.step()
        test_epoch_log = test(test_loader, model, criterion, args)

        acc1 = test_epoch_log[1]
        is_best = acc1 > best_prec1

        best_prec1 = max(acc1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
         }, is_best, filepath=model_dir)
    print("Best accuracy: "+str(best_prec1))
    end_time = time.time()
    print("Total Time:", end_time - start_time)
    with open(os.path.join(model_dir, 'BestAcc_TotalTime_' + '.txt'), 'w') as weights_log_file:
        with redirect_stdout(weights_log_file):
            print("Best accuracy: "+str(best_prec1))
            print("Total Time:", end_time - start_time)
if __name__ == '__main__':
    main()
