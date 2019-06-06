import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import argparse
import random
import pickle
import scipy.misc

import models.videos.model_simple as models

from opts import parse_opts
from geotnf.transformation import GeometricTnf

from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset_utils import Logger
from datasets.hmdb51 import HMDB51
from train import train_epoch
from validation import val_epoch
import test
import eval_hmdb51



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

    print('\n')

    params['predDistance'] = state['predDistance']
    print('predDistance: ' + str(params['predDistance']))

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


if __name__ == '__main__':

    opt = parse_opts()

    print("Gpu ID's:", opt.gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    print("Torch version:", torch.__version__)
    print("Train, val, test, evaluate:", not opt.no_train, opt.val, not opt.no_test, not opt.no_eval)

    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        split_list = opt.list.split("_")[1][0]
        split_annotation = opt.annotation_path.split("_")[1][0]
        if split_list != split_annotation:
            print("Please provide list and annotation for same split")
            exit()
        split = (opt.annotation_path.split(".")[0]).split("/")[-1]
        print("Split of HMDB51:", split)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.list = os.path.join(opt.root_path, opt.list)
        folder = opt.result_path
        opt.result_path = os.path.join(opt.root_path, opt.result_path + "_" + split)
        if not os.path.isdir(opt.result_path):
            os.mkdir(opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    if opt.val and opt.test_eval_all:
        print("Choose validation with test and eval, or other validation, not both")
        exit()

    params, state = get_params(opt)

    print("Result path:", opt.result_path)
    print("Resume path:", opt.resume_path)
    print("Video path:", opt.video_path)
    print("Annotation path:", opt.annotation_path) 

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    print("Architecture:", opt.arch)

    # Random seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if not opt.no_cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    print("\n")

    print("Sample Size:", opt.sample_size)
    print("Video Len:", opt.videoLen)
    print("Frame Gap:", opt.frame_gap)
    print("Pred Distance:", opt.predDistance)
    print("Sample Duration:", opt.sample_duration)
    print("TimeCycle weight:", opt.timecycle_weight)
    print("Binary classification weight:", opt.binary_class_weight)

    model = models.CycleTime(class_num=params['n_classes'], 
                             trans_param_num=3, 
                             frame_gap=opt.frame_gap,
                             videoLen=opt.videoLen,
                             sample_duration=opt.sample_duration,
                             pretrained=opt.pretrained_imagenet, 
                             temporal_out=params['videoLen'], 
                             T=opt.T, 
                             hist=opt.hist)

    if not opt.no_cuda:
        model = model.cuda()

    cudnn.benchmark = False
    print(' Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    print('Weight_decay: ' + str(opt.weight_decay))
    print('Beta1: ' + str(opt.momentum))

    print("\n")
    print("LOADING PRETRAIN/RESUME AND LOGGER")
    print("\n")

    
    optimizer = optim.Adam(model.parameters(), 
                lr=opt.learning_rate, 
                betas=(opt.momentum, 0.999), 
                weight_decay=opt.weight_decay)

    print("\n")
    print("Adam Optimizer made")

    if opt.pretrain_path:
        # Load checkpoint.
        print('Loading pretrained model {}'.format(opt.pretrain_path))
        assert os.path.isfile(opt.pretrain_path), 'No pretrain directory found'
        checkpoint = torch.load(opt.pretrain_path)

        partial_load(checkpoint['state_dict'], model)

        del checkpoint

    if opt.resume_path:
        # Load checkpoint.
        print('Loading checkpoint {}'.format(opt.resume_path))
        assert os.path.isfile(opt.resume_path), 'No checkpoint directory found'
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']
        opt.begin_epoch = checkpoint['epoch']

        partial_load(checkpoint['state_dict'], model)
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

        train_log_file = 'train_resume_{}.log'.format(opt.begin_epoch)
        train_batch_log_file = 'train_batch_resume_{}.log'.format(opt.begin_epoch)
        val_log_file = 'val_resume_{}.log'.format(opt.begin_epoch)

        opts_file = os.path.join(opt.result_path, 'opts_resume_{}.json'.format(opt.begin_epoch))
        
        del checkpoint

    else:

        train_log_file = 'train.log'
        train_batch_log_file = 'train_batch.log'
        val_log_file = 'val.log'

        opts_file = os.path.join(opt.result_path, 'opts.json')

      

    if not opt.no_train:

        # Save opts

        print("\n")
        print("Save opts at", opts_file)
        with open(opts_file, 'w') as opt_file:
            json.dump(vars(opt), opt_file)


        print("\n")
        print("TRAINING")
        print("\n")

        train_logger = Logger(
           os.path.join(opt.result_path, train_log_file),
           ['epoch', 'loss', 'loss_hmdb_class', 'loss_timecycle', 'loss_bin_class', 'acc', 'acc_bin', 'lr', 'loss_sim', 'theta_loss', 'theta_skip_loss'])
        train_batch_logger = Logger(
           os.path.join(opt.result_path, train_batch_log_file),
           ['epoch', 'batch', 'iter', 'loss_hmdb_class', 'loss_timecycle', 'loss_bin_class', 'acc', 'acc_bin', 'lr', 'loss_sim', 'theta_loss', 'theta_skip_loss'])
        
        target_transform = ClassLabel()

        geometric_transform = GeometricTnf(
            'affine', 
            out_h=params['cropSize2'], 
            out_w=params['cropSize2'], 
            use_cuda = False)

        training_data = HMDB51(
            params,
            opt.video_path,
            opt.annotation_path,
            'training',
            frame_gap=opt.frame_gap,
            sample_duration=opt.sample_duration,
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
        print("Learning rate:", opt.learning_rate)
        print("Momentum:", opt.momentum)
        print("Weight decay:", opt.weight_decay)

        scheduler = lr_scheduler.ReduceLROnPlateau(
           optimizer, 
           'min', 
           patience=opt.lr_patience)

        print("Lr_patience", opt.lr_patience)

        print("\n")


    if opt.val or opt.test_eval_all:

        print("VALIDATION")
        print("\n")

        val_logger = Logger(
            os.path.join(opt.result_path, val_log_file), ['epoch', 'loss', 'acc'])

        target_transform = ClassLabel()

        if opt.val:
            n = opt.n_val_samples

        if opt.test_eval_all:
            n = 0

        validation_data = HMDB51(
            params,
            opt.video_path,
            opt.annotation_path,
            'validation',
            sample_duration=opt.sample_duration,
            n_samples_for_each_video=opt.n_val_samples,
            target_transform=target_transform)

        print("Validation data loaded")

        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)

        print("Validation loader done")

    

    #print("MODEL:", model.state_dict().keys())

    print("\n")
    print("RUNNING")
    print("\n")


    for i in range(opt.begin_epoch, opt.n_epochs + 1):


        if not opt.no_train:

            train_epoch(i, params, train_loader, model, criterion, optimizer, opt, train_logger, train_batch_logger)

        # if opt.test_eval_all:

        #     # if i % opt.checkpoint == 0:
        #     validation_loss, file_json = val_test_eval_epoch(i, params, val_loader, model, criterion, opt, val_logger, validation_data.class_names)
        #     name_general = opt.result_path + '/' + "results" + '_all' + '.txt'
        #     name = opt.result_path + '/' + "results_" + str(i) + '.txt'
        #     eval_hmdb51.eval_hmdb51(name, opt.annotation_path, file_json, "validation", opt.top_k,  name_general, i)

        # if not opt.no_train and opt.test_eval_all:

        #     # if i % opt.checkpoint == 0:
        #     scheduler.step(validation_loss)

        if opt.val:

            validation_loss = val_epoch(i, params, val_loader, model, criterion, opt, val_logger)

        if not opt.no_train and opt.val:

            scheduler.step(validation_loss)


    if not opt.no_test:

        print("\n")
        print("TESTING")

        target_transform = VideoID()

        test_data =  HMDB51(
            params,
            opt.video_path,
            opt.annotation_path,
            "validation",
            sample_duration=opt.sample_duration,
            n_samples_for_each_video=0,
            target_transform=target_transform)

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        
        if opt.general_eval_file:
            val_json_name = 'temp'
        else:
            if not opt.no_train and opt.val:
                epoch = opt.n_epochs
            else:
                epoch = opt.begin_epoch - 1
            val_json_name = str(epoch)

        test.test(test_loader, model, opt, test_data.class_names, val_json_name)

    if not opt.no_eval:

        print("\n")
        print("EVALUATING")

        if not opt.no_train and opt.val:
            epoch = opt.n_epochs

        else:
            epoch = opt.begin_epoch - 1
            
        eval_path = opt.result_path + '/' + "results" + '_' + str(epoch) + '.txt'
        
        print("File:", eval_path)

        prediction_file = os.path.join(opt.result_path, 'val_{}.json'.format(val_json_name))
        general_output_path = os.path.join(opt.result_path, 'results_all.txt')
        subset = "validation"

        if opt.general_eval_file:
            f = open(general_output_path, 'w')
            f.write("Results \n")
            f.write("Epoch \t Accuracy \t Error \n")
            f.close()

        eval_hmdb51.eval_hmdb51(eval_path, opt.annotation_path, prediction_file, subset, opt.top_k, general_output_path, epoch, opt.general_eval_file)
