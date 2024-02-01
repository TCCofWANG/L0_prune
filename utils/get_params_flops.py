import torch
import torch.nn as nn
import argparse
from importlib import import_module
from thop import profile
import models as models

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--data_set',type=str,default='cifar10',)
parser.add_argument('--gpus',type=int,nargs='+',default=[0],help='Select gpu_id to use. default:[0]')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--vgg_arch', type = str, default = 'vgg_cifar', help = 'Architecture of model. default:vgg_cifar')
parser.add_argument('--resnet_arch', type = str, default = 'resnet_cifar', help = 'Architecture of model. default:resnet_cifar')
parser.add_argument('--cfg', type = str, default = 'vgg16', help = 'Detail architecuture of model. default:vgg16')
parser.add_argument('--honey',type=str,default=None,help='The prune rate of CNN guided by best honey (or layer_cfg for vgg19 and resnet50)')
args = parser.parse_args()

device = torch.device("cuda:0")

print('==> Building model..')
if args.arch == 'vgg':
    if args.vgg_arch == 'vgg_cifar':
        model_base = import_module(f'models.{args.vgg_arch}').BeeVGG(args.cfg,honeysource = [0] * 13).to(device)
        model_prune = import_module(f'models.{args.vgg_arch}').BeeVGG(args.cfg,args.honey).to(device)
    else:
        model_base = models.__dict__[args.arch](depth=19).to(device)
        model_prune = model = models.__dict__[args.arch](depth=19, cfg = checkpoint['cfg']).to(device)
elif args.arch == 'resnet':
    if args.resnet_arch == 'resnet_cifar':
        model_base = import_module(f'models.{args.resnet_arch}').resnet(args.cfg).to(device)
        model_prune = import_module(f'models.{args.resnet_arch}').resnet(args.cfg, honey = args.honey).to(device)
    elif args.resnet_arch == 'resnet':
        model_base = import_module(f'models.{args.resnet_arch}').resnet(args.cfg).to(device)
        if args.honey == None:
            model_prune = import_module(f'models.{args.resnet_arch}').resnet50_flops(layer_cfg = args.honey).to(device)
        else:
            model_prune = import_module(f'models.{args.resnet_arch}').resnet(args.cfg, honey = args.honey).to(device)

if args.data_set == 'cifar10':
    input_image_size = 32
elif args.data_set == 'imagenet':
    input_image_size = 224

input = torch.randn(1, 3, input_image_size, input_image_size).to(device)

baseflops, baseparams = profile(model_base, inputs=(input, ))
pruflops, pruparams = profile(model_prune, inputs=(input, ))


print('--------------Base Model--------------')
print('Params: %.2f M '%(baseparams/1000000))
print('FLOPS: %.2f M '%(baseflops/1000000))

print('--------------Pruned Model--------------')
print('Params: %.2f M '%(pruparams/1000000))
print('FLOPS: %.2f M '%(pruflops/1000000))

