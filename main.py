# TimeCycle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import argparse
import os
import shutil
import time
import random

import numpy as np
import pickle
import scipy.misc

import utils.imutils2
import models.videos.model_simple as models
# from utils import Logger, AverageMeter, savefig

import models.dataset.vlog_train as vlog

from opts import parse_opts

from geotnf.transformation import GeometricTnf

# 3D-ResNets

import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset_utils import Logger
from datasets.hmdb51 import HMDB51
from train import train_epoch
from validation import val_epoch
import test
import eval_hmdb51

import os

def partial_load(pretrained_dict, model):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)

def get_params(opt):

    params = {}
    params['filelist'] = opt.list
    params['imgSize'] = 256
    params['imgSize2'] = 320
    params['cropSize'] = 240
    params['cropSize2'] = 80
    params['offset'] = 0

    state = {k: v for k, v in opt._get_kwargs()}

    params['predDistance'] = state['predDistance']
    print(params['predDistance'])

    params['batch_size'] = state['batch_size']
    print('batch_size: ' + str(params['batch_size']) )

    print('temperature: ' + str(state['T']))

    params['gridSize'] = state['gridSize']
    print('gridSize: ' + str(params['gridSize']) )

    params['n_classes'] = state['n_classes']
    print('n_classes: ' + str(params['n_classes']) )

    params['videoLen'] = state['videoLen']
    print('videoLen: ' + str(params['videoLen']) )

    return params, state

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth'):
    new_model_state = {}
    model_state = state['state_dict']
    
    for key in model_state.keys():
        if "encoderVideo" in key:
            new_model_state[key.replace("encoderVideo.", "")] = model_state[key]
        else:
            new_model_state[key] = model_state[key]

    state['state_dict'] = new_model_state

    epoch = state['epoch']
    filename = 'checkpoint_' + str(epoch) + '.pth'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()


if __name__ == '__main__':

    global best_loss

    opt = parse_opts()

    print("Gpu ID's:", opt.gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    print("Torch version:", torch.__version__)
    print("Train, val, test, evaluate:", not opt.no_train, not opt.no_val, not opt.no_test, not opt.no_eval)

    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.list = os.path.join(opt.root_path, opt.list)
        split = opt.annotation_path.split(".")[0]
        print("SPLIT:", split)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        folder = opt.result_path
        opt.result_path = os.path.join(opt.root_path, opt.result_path + "_" + split)
        print("Result path:", opt.result_path)
        opt.path_checkpoint = os.path.join(opt.root_path, opt.path_checkpoint + "_" + split)
        print("Path checkpoint:", opt.path_checkpoint)
        if not os.path.isdir(opt.result_path):
            os.mkdir(opt.result_path)
        if not os.path.isdir(opt.path_checkpoint):
            os.mkdir(opt.path_checkpoint)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    params, state = get_params(opt)

    print("Result path:", opt.result_path)
    print("Checkpoint path:", opt.result_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print("Architecture:", opt.arch)
    print("Opt.mean", opt.mean)
    print("opt.std", opt.std)
    
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    # Random seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if not opt.no_cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    best_loss = 0  # best test accuracy

    model = models.CycleTime(class_num=params['n_classes'], 
                             trans_param_num=3, 
                             pretrained=opt.pretrained_imagenet, 
                             temporal_out=params['videoLen'], 
                             T=opt.T, 
                             hist=opt.hist)

    if not opt.no_cuda:
        model = model.cuda()

    cudnn.benchmark = False
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
        print("Norm method 0")
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
        print("Norm method 1")
    else:
        norm_method = Normalize(opt.mean, opt.std)
        print("Norm method 2")

    print('Weight_decay: ' + str(opt.u_wd))
    print('Beta1: ' + str(opt.u_momentum))

    print("\n")
    print("LOADING PRETRAIN/RESUME AND LOGGER")
    print("\n")

    if opt.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                    lr=opt.u_lr, 
                    betas=(opt.u_momentum, 0.999), 
                    weight_decay=opt.u_wd)
        print("Adam")
    else:
        optimizer = optim.SGD(model.parameters(), 
                          lr=opt.u_lr, 
                          weight_decay=opt.u_wd, 
                          momentum=0.95
                          #dampening=0.9,
                          #nesterov=False
                          )
        print("SGD")

    print("\n")
    print("Optimizer made")

    if opt.pretrain_path:
        # Load checkpoint.
        print('Loading pretrained model {}'.format(opt.pretrain_path))
        assert os.path.isfile(opt.pretrain_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.pretrain_path)

        partial_load(checkpoint['state_dict'], model)

        del checkpoint

    title = 'videonet'
    if opt.resume_path:
        # Load checkpoint.
        print('Loading checkpoint {}'.format(opt.resume_path))
        assert os.path.isfile(opt.resume_path), 'Error: no checkpoint directory found!'
        opt.path_checkpoint = os.path.dirname(opt.resume_path)
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']
        opt.begin_epoch = checkpoint['epoch']

        partial_load(checkpoint['state_dict'], model)
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

        # logger = Logger(os.path.join(opt.path_checkpoint, 'log-resume.txt'), title=title)
        # logger.set_names(['Learning Rate', 'Train Loss', 'Theta Loss', 'Theta Skip Loss'])


        del checkpoint
    

        
        # logger = Logger(os.path.join(opt.path_checkpoint, 'log.txt'), title=title)
        # logger.set_names(['Learning Rate', 'Train Loss', 'Theta Loss', 'Theta Skip Loss'])

  

    if not opt.no_train:

        print("\n")
        print("TRAINING")
        print("\n")


        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])

        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value), norm_method
        ])

        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()

        geometric_transform = GeometricTnf(
            'affine', 
            out_h=params['cropSize2'], 
            out_w=params['cropSize2'], 
            use_cuda = False)

        print("PARAMS:", type(params))
        training_data = HMDB51(
            params,
            opt.video_path,
            opt.annotation_path,
            'training',
            frame_gap=opt.frame_gap,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            geometric_transform=geometric_transform)

        print("Training data obtained")
        
        train_loader = torch.utils.data.DataLoader(
           training_data,
           batch_size=opt.batch_size,
           shuffle=True,
           num_workers=opt.n_threads,
           pin_memory=True)

        print("Train loader made")


        train_logger = Logger(
           os.path.join(opt.result_path, 'train.log'),
           ['epoch', 'loss', 'loss_vc', 'loss_main', 'acc', 'lr'])
        train_batch_logger = Logger(
           os.path.join(opt.result_path, 'train_batch.log'),
           ['epoch', 'batch', 'iter', 'loss', 'loss_vc', 'loss_main', 'acc', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening

        print("Learning rate:", opt.learning_rate)
        print("Momentum:", opt.momentum)
        print("Dampening:", opt.dampening)
        print("Weight decay:", opt.weight_decay)
        print("Nesterov:", opt.nesterov)

        scheduler = lr_scheduler.ReduceLROnPlateau(
           optimizer, 
           'min', 
           patience=opt.lr_patience)

        print("\n")
        print("Lr_patience", opt.lr_patience)

        print("\n")


    if not opt.no_val:

        print("VALIDATION")
        print("\n")

        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])

        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()

        geometric_transform = GeometricTnf(
            'affine', 
            out_h=params['cropSize2'], 
            out_w=params['cropSize2'], 
            use_cuda = False)

        validation_data = HMDB51(
            params,
            opt.video_path,
            opt.annotation_path,
            'validation',
            frame_gap=opt.frame_gap,
            n_samples_for_each_video=opt.n_val_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            geometric_transform=geometric_transform,
            sample_duration=opt.sample_duration)

        print("Validation data loaded")

        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)

        print("Validation loader done")


        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    #print("MODEL:", model.state_dict().keys())

    print("\n")
    print("RUNNING")
    print("\n")

    for i in range(opt.begin_epoch, opt.n_epochs + 1):


        if not opt.no_train:


            #print("Train epoch")

            loss, acc, main_loss, losses_theta, losses_theta_skip = train_epoch(i, params, train_loader, model, criterion, optimizer, opt, train_logger, train_batch_logger)


        if not opt.no_val:

            #print("Val epoch")

            validation_loss = val_epoch(i, params, val_loader, model, criterion, opt,
                                        val_logger)

        if not opt.no_train and not opt.no_val:

            #print("Lr schedule")

            scheduler.step(validation_loss)


    if not opt.no_test:

        print("TESTING")

        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
        ])

        temporal_transform = LoopPadding(opt.sample_duration)

        target_transform = VideoID()

        test_data =  HMDB51(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        
        test.test(test_loader, model, opt, test_data.class_names)

    if not opt.no_eval:

        #print("EVALUATING")

        name = opt.result_path + '/' + "results" + '_' + str(opt.n_epochs) + '.txt'
        #print("File:", name)

        prediction = os.path.join(opt.result_path, "val.json")
        subset = "validation"

        eval_hmdb51.eval_hmdb51(name, opt.annotation_path, prediction, subset, 1)


