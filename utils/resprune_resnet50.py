import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
from importlib import import_module

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--dataset', type=str, default='imagenet',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=50,
                    help='depth of the resnet')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./prune_result', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--resnet_arch', type = str, default = 'resnet', help = 'Architecture of model. default:vgg_cifar')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
if not os.path.exists(args.save):
    os.makedirs(args.save)

model = import_module(f'models.{args.resnet_arch}').resnet50_flops().to(device)

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model,map_location = device)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}" .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

prunemodel_save_name = ""
model_dir = os.path.join(os.path.join(os.path.join(os.getcwd(), "prune_result"), args.dataset), prunemodel_save_name)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir, exist_ok=True)
savepath = os.path.join(model_dir, "Pre_prune_model_parameters.txt")
#======================================================================
cfg = []
layer_cfg = []
cfg_mask = []
bn_first_layer = 0
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        size = m.weight.data.shape[0]
        mask = weight_copy.gt(0).float().to(device)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg_mask.append(mask.clone())

for name,module in model.named_modules():
    if "bn1" in name or "bn2" in name:
        if bn_first_layer == 0:
            bn_first_layer += 1
            continue
        else:
            w = module.weight.data.abs().clone()
            layer_cfg.append(torch.nonzero(w).size(0))

with open(savepath, "w") as fp:
    fp.write("Configuration(cfg): "+str(layer_cfg)+"\n")
# =======================================================================
newmodel = import_module(f'cifar10_models.{args.resnet_arch}').resnet50_flops(layer_cfg = layer_cfg).to(device)
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
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        if conv_count == 4 or conv_count == 14 or conv_count == 27 or conv_count == 46:
            layer_id_in_cfg += 1
        else:
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        if conv_count == 0 or conv_count == 4 or conv_count == 14 or conv_count == 27 or conv_count == 46:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        if isinstance(old_modules[layer_id+1], nn.BatchNorm2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            conv_count += 1
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': layer_cfg, 'state_dict': newmodel.state_dict()}, os.path.join(model_dir, 'pruned.pth.tar'))
