from __future__ import print_function
import os
import datetime
import csv
import argparse
import random
import shutil
import numpy as np
import time
import math
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
import random
import heapq
import copy
import time
from importlib import import_module
# Training settings
parser = argparse.ArgumentParser(description = 'PyTorch')
parser.add_argument('--dataset', type = str, default = 'cifar10',
                    help = 'training dataset (default: cifar10)')
parser.add_argument('-j', '--workers', default = 4, type = int, metavar = 'N',
                    help = 'number of data loading workers (default: 4)')
parser.add_argument('--refine', default = '', type = str, metavar = 'PATH',
                    help = 'path to the pruned model to be fine tuned')
parser.add_argument('-opt', '--optimizer', metavar = 'OPT', default = "SGD",
                    help = 'optimizer algorithm')
parser.add_argument('--batch-size', type = int, default = 64, metavar = 'N',
                    help = 'input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type = int, default = 256, metavar = 'N',
                    help = 'input batch size for testing (default: 256)')
parser.add_argument('--epochs', type = int, default = 160, metavar = 'N',
                    help = 'number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default = 0, type = int, metavar = 'N',
                    help = 'manual epoch number (useful on restarts)')
parser.add_argument('--lr', type = float, default = 0.1, metavar = 'LR',
                    help = 'learning rate (default: 0.1)')
parser.add_argument('--momentum', type = float, default = 0.9, metavar = 'M',
                    help = 'SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default = 0.0001, type = float,
                    metavar = 'W', help = 'weight decay (default: 1e-4)')
parser.add_argument('--resume', default = ''
                    , type = str, metavar = 'PATH',
                    help = 'path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action = 'store_true', default = False,
                    help = 'disables CUDA training')
parser.add_argument('-p', '--print-freq', default = 100, type = int,
                    metavar = 'N', help = 'print frequency (default: 50)')
parser.add_argument('--save-model', default = True, type = lambda x: bool(distutils.util.strtobool(x)),
                    help = 'For Saving the current Model (default: True)')
# ----------------------------------------------------------------------------------------------------
parser.add_argument('--max_cycle', type = int, default = 10, help = 'Search for best pruning plan times. default:10')
parser.add_argument('--min_prune', type = int, default =2, help = 'Minimum percent of prune per layer')
parser.add_argument('--food_number', type = int, default = 10, help = 'Food number')
parser.add_argument('--food_dimension', type = int, default = 13,
                    help = 'Food dimension: num of conv layers. default: vgg16->13(resnet56-> 27,resnet110-> 54)conv layer to be pruned')
parser.add_argument('--food_limit', type = int, default =2,
                    help = 'Beyond this limit, the bee has not been renewed to become a scout bee. default:2')
parser.add_argument('--honeychange_num', type = int, default =5,
                    help = 'Number of codes that the nectar source changes each time')
parser.add_argument('--honey_model', type = str, default = './pretrain/vgg16_cifar10.pt',
                    help = 'Pretrained model. default:None')
parser.add_argument('--arch', type = str, default = 'vgg_cifar', help = 'Architecture of model. default:vgg_cifar')
parser.add_argument('--cfg', type = str, default = 'vgg16', help = 'Detail architecuture of model. default:vgg16')
# ----------------------------------------------------------------------------------------------------
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

criterion = torch.nn.CrossEntropyLoss()
device = torch.device("cuda:1" if use_cuda else "cpu")
kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}

if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root = './data/cifar10', train = True, download = True,
                             transform = transforms.Compose([
                                 transforms.Pad(4),
                                 transforms.RandomCrop(32),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                 ])),
            batch_size = args.batch_size, shuffle = True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root = './data/cifar10', train = False, download = True,
                             transform = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                 ])),
            batch_size = args.test_batch_size, shuffle = False, **kwargs)
elif args.dataset == 'cifar100':
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root = './data/cifar100', train = True, download = True,
                              transform = transforms.Compose([
                                  transforms.Pad(4),
                                  transforms.RandomCrop(32),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                  ])),
            batch_size = args.batch_size, shuffle = True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root = './data/cifar100', train = False, download = True, transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])),
            batch_size = args.test_batch_size, shuffle = False, **kwargs)


print("=> creating model '{}'".format(args.arch))
if args.arch == 'vgg_cifar':
    bee_model = import_module(f'models.{args.arch}').VGG(args.cfg).to(device)
    ckpt_pre = torch.load(args.honey_model, map_location = device)
    bee_model.load_state_dict(ckpt_pre['state_dict'])
elif args.arch == 'resnet_cifar':
    bee_model = import_module(f'models.{args.resnet_arch}').resnet(args.cfg).to(device)
    ckpt_pre = torch.load(args.honey_model, map_location = device)
    bee_model.load_state_dict(ckpt_pre['state_dict'])


current_name = ""
model_name = '%s%s' % (args.arch, current_name)

if (args.save_model):
    model_dir = os.path.join(os.path.join(os.path.join(os.getcwd(), "../models"), args.dataset), model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok = True)


class BeeGroup():
    def __init__(self):
        super(BeeGroup, self).__init__()
        self.code = []
        self.fitness = 0
        self.rfitness = 0
        self.trail = 0

best_honey = BeeGroup()
NectraSource = []
EmployedBee = []
OnLooker = []
best_honey_state = {}

# Calculate fitness of a honey source
def calculationFitness(honey, args):
    global best_honey
    global best_honey_state

    if args.arch == 'vgg_cifar':
        model = import_module(f'models.{args.arch}').VGG(args.cfg).to(device)
        ckpt_pre = torch.load(args.honey_model, map_location = device)
        model.load_state_dict(ckpt_pre['state_dict'])
    elif args.arch == 'resnet_cifar':
        model = import_module(f'models.{args.arch}').resnet(args.cfg).to(device)
        ckpt_pre = torch.load(args.honey_model, map_location = device)
        model.load_state_dict(ckpt_pre['state_dict'])

    fit_accurary = AverageMeter('Acc@1', ':.2f')

    model.train()

    if args.arch == 'vgg_cifar':
        gamma_index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                y = m.weight.data
                y_abs = torch.abs(y).to(device)
                mask = y_abs.gt(0).float()
                nonzero_index = int(torch.sum(mask))
                size = m.weight.data.shape[0]
                if nonzero_index > int((1 - honey[gamma_index] / 10) * size):
                    y_hat = projection(y, size, int((1 - honey[gamma_index] / 10) * size)).to(device)
                    m.weight.data = y_hat
                gamma_index = gamma_index + 1
    elif args.arch == 'resnet_cifar':
        gamma_index = 0
        flag = 1
        bn_first_layer = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                if bn_first_layer == 0:
                    bn_first_layer = bn_first_layer + 1
                    continue
                if flag % 2 == 1:
                    y = m.weight.data
                    y_abs = torch.abs(y).to(device)
                    mask = y_abs.gt(0).float()
                    nonzero_index = int(torch.sum(mask))
                    size = m.weight.data.shape[0]
                    if nonzero_index > int((1 - honey[gamma_index] / 10) * size):
                        y_hat = projection(y, size, int((1 - honey[gamma_index] / 10) * size)).to(device)
                        m.weight.data = y_hat
                    gamma_index = gamma_index + 1
                    flag = flag + 1
                else:
                    flag = flag + 1

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            acc1,acc5 = accuracy(outputs, targets,topk = (1,5))
            fit_accurary.update(acc1[0], inputs.size(0))

    if fit_accurary.avg > best_honey.fitness:
        best_honey_state = copy.deepcopy(model.state_dict())
        best_honey.code = copy.deepcopy(honey)
        best_honey.fitness = fit_accurary.avg

    return fit_accurary.avg

#Initilize
def initilize():
    print('==> Initilizing Honey_model..')
    global best_honey, NectraSource, EmployedBee, OnLooker

    for i in range(args.food_number):
        NectraSource.append(copy.deepcopy(BeeGroup()))
        EmployedBee.append(copy.deepcopy(BeeGroup()))
        OnLooker.append(copy.deepcopy(BeeGroup()))
        for j in range(args.food_dimension):
            NectraSource[i].code.append(copy.deepcopy(random.randint(args.min_prune,9)))

        # initilize honey souce
        NectraSource[i].fitness = calculationFitness(NectraSource[i].code, args)
        NectraSource[i].rfitness = 0
        NectraSource[i].trail = 0

        # initilize employed bee
        EmployedBee[i].code = copy.deepcopy(NectraSource[i].code)
        EmployedBee[i].fitness = NectraSource[i].fitness
        EmployedBee[i].rfitness = NectraSource[i].rfitness
        EmployedBee[i].trail = NectraSource[i].trail

        # initilize onlooker
        OnLooker[i].code = copy.deepcopy(NectraSource[i].code)
        OnLooker[i].fitness = NectraSource[i].fitness
        OnLooker[i].rfitness = NectraSource[i].rfitness
        OnLooker[i].trail = NectraSource[i].trail

    # # initilize best honey
    # best_honey.code = copy.deepcopy(NectraSource[0].code)
    # best_honey.fitness = NectraSource[0].fitness
    # best_honey.rfitness = NectraSource[0].rfitness
    # best_honey.trail = NectraSource[0].trail

#Employed Bees
def sendEmployedBees():
    global NectraSource, EmployedBee
    for i in range(args.food_number):
        while 1:
            k = random.randint(0, args.food_number - 1)
            if k != i:
                break

        EmployedBee[i].code = copy.deepcopy(NectraSource[i].code)
        param2change = np.random.randint(0, args.food_dimension -1, args.honeychange_num)
        R = np.random.uniform(-1, 1, args.honeychange_num)
        for j in range(args.honeychange_num):
            EmployedBee[i].code[param2change[j]] = int(NectraSource[i].code[param2change[j]] + R[j] * (
                    NectraSource[i].code[param2change[j]] - NectraSource[k].code[param2change[j]]))
            if EmployedBee[i].code[param2change[j]] > 9:
                EmployedBee[i].code[param2change[j]] = 9
            if EmployedBee[i].code[param2change[j]] < args.min_prune:
                EmployedBee[i].code[param2change[j]] = args.min_prune

        EmployedBee[i].fitness = calculationFitness(EmployedBee[i].code, args)

        if EmployedBee[i].fitness > NectraSource[i].fitness:
            NectraSource[i].code = copy.deepcopy(EmployedBee[i].code)
            NectraSource[i].trail = 0
            NectraSource[i].fitness = EmployedBee[i].fitness
        else:
            NectraSource[i].trail = NectraSource[i].trail + 1

def calculateProbabilities():
    global NectraSource

    maxfit = NectraSource[0].fitness

    for i in range(1, args.food_number):
        if NectraSource[i].fitness > maxfit:
            maxfit = NectraSource[i].fitness
    for i in range(args.food_number):
        NectraSource[i].rfitness = (0.9 * (NectraSource[i].fitness / maxfit)) + 0.1

#Onlooker Bees
def sendOnlookerBees():
    global NectraSource, EmployedBee, OnLooker
    i = 0
    t = 0
    while t < args.food_number:
        R_choosed = random.uniform(0, 1)
        if (R_choosed <= NectraSource[i].rfitness):
            t += 1
            while 1:
                k = random.randint(0, args.food_number-1)
                if k != i:
                    break
            OnLooker[i].code = copy.deepcopy(NectraSource[i].code)
            param2change = np.random.randint(0, args.food_dimension -1, args.honeychange_num)
            R = np.random.uniform(-1, 1, args.honeychange_num)
            for j in range(args.honeychange_num):
                OnLooker[i].code[param2change[j]] = int(NectraSource[i].code[param2change[j]] + R[j] * (
                        NectraSource[i].code[param2change[j]] - NectraSource[k].code[param2change[j]]))
                if OnLooker[i].code[param2change[j]] > 9:
                    OnLooker[i].code[param2change[j]] = 9
                if OnLooker[i].code[param2change[j]] < args.min_prune:
                    OnLooker[i].code[param2change[j]] = args.min_prune

            OnLooker[i].fitness = calculationFitness(OnLooker[i].code, args)

            if OnLooker[i].fitness > NectraSource[i].fitness:
                NectraSource[i].code = copy.deepcopy(OnLooker[i].code)
                NectraSource[i].trail = 0
                NectraSource[i].fitness = OnLooker[i].fitness
            else:
                NectraSource[i].trail = NectraSource[i].trail + 1
        i += 1
        if i == args.food_number:
            i = 0

#Scout Bees
def sendScoutBees():
    global NectraSource, EmployedBee, OnLooker
    for i in range(args.food_number):
        if NectraSource[i].trail >= args.food_limit:
            for j in range(args.food_dimension):
                NectraSource[i].code[j] = random.randint(args.min_prune,9)
            NectraSource[i].trail = 0
            NectraSource[i].fitness = calculationFitness(NectraSource[i].code, args)

def memorizeBestSource():
    global best_honey, NectraSource
    for i in range(args.food_number):
        if NectraSource[i].fitness > best_honey.fitness:
            best_honey.code = copy.deepcopy(NectraSource[i].code)
            best_honey.fitness = NectraSource[i].fitness

def projection(x, size, size_x):
    y = torch.zeros(size).to(device)
    x_max = torch.abs(x).to(device)
    values, indices = torch.topk(x_max, k = size_x)
    for index, value in enumerate(indices.to(device)):
        y[value] = x[value]
    return y

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
    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk = (1, 5))
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
    batch_time = AverageMeter('Time', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':.2f')
    top5 = AverageMeter('Acc@5', ':.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, prefix = 'Test: ')
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk = (1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'.format(top1 = top1))

    return (losses.avg, top1.avg.cpu().numpy(), batch_time.avg, best_honey.code)

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix = ""):
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

def accuracy(output, target, topk = (1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim = True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

test(test_loader, bee_model, criterion, args)
initilize()
# memorizeBestSource()
print('===> Start BeePruning..')
start_time = time.time()
for cycle in range(args.max_cycle):
    current_time = time.time()
    sendEmployedBees()
    calculateProbabilities()
    sendOnlookerBees()
    # memorizeBestSource()
    sendScoutBees()
    # memorizeBestSource()
    print(
            'Search Cycle [{}]\t Best Honey Source {}\tBest Honey Source fitness {:.2f}%\tTime {:.2f}s\n'
                .format(cycle, best_honey.code, float(best_honey.fitness), (current_time - start_time))
            )
print('===> BeePruning Complete!')

with open(os.path.join(model_dir, "train_test_log.csv"), "w") as train_log_file:
    train_log_csv = csv.writer(train_log_file)
    train_log_csv.writerow(
            ['epoch', 'train_loss', 'train_top1_acc', 'train_time', 'test_loss', 'test_top1_acc', 'test_time',
             'prune_radio'])
start_time = time.time()
best_prec1 = 0.

if args.arch == 'vgg_cifar':
    model = import_module(f'models.{args.arch}').VGG(args.cfg).to(device)
    model.load_state_dict(best_honey_state)
elif args.arch == 'resnet_cifar':
    model = import_module(f'models.{args.arch}').resnet(args.cfg).to(device)
    model.load_state_dict(best_honey_state)

cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(0).float().to(device)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())

#Generate new model
if args.arch == 'vgg_cifar':
    newmodel = import_module(f'models.{args.vgg_arch}').BeeVGG(args.cfg,best_honey.code).to(device)

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            w2 = m0.bias.data[idx1.tolist()].clone()
            m1.weight.data = w1.clone()
            m1.bias.data = w2.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
elif args.arch == 'resnet_cifar':
    newmodel = import_module(f'models.{args.resnet_arch}').resnet(args.cfg, best_honey.code).to(device)
    old_modules = list(model.modules())
    new_modules = list(newmodel.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0
    bn_first_layer = 0
    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if isinstance(old_modules[layer_id + 1], nn.BatchNorm2d):
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

optimizer = torch.optim.SGD(newmodel.parameters(), args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [80,120], gamma = 0.1)

for epoch in range(args.start_epoch, args.epochs):
    train_epoch_log = train(train_loader, newmodel, criterion, optimizer, epoch, args)
    scheduler.step()
    test_epoch_log = test(test_loader, newmodel, criterion, args)
    acc1 = test_epoch_log[1]

    with open(os.path.join(model_dir, "train_test_log.csv"), "a") as train_log_file:
        train_log_csv = csv.writer(train_log_file)
        train_log_csv.writerow(((epoch,) + train_epoch_log + test_epoch_log))

    is_best = acc1 > best_prec1
    best_prec1 = max(acc1, best_prec1)

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': newmodel.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
        }, is_best, filepath = model_dir)
    # ===============================================================================
print("Best accuracy: " + str(best_prec1))
end_time = time.time()
print("Total Time:", end_time - start_time)
with open(os.path.join(model_dir, 'BestAcc_TotalTime_' + '.txt'), 'w') as weights_log_file:
    with redirect_stdout(weights_log_file):
        print("Best accuracy: " + str(best_prec1))
        print("Total Time:", end_time - start_time)