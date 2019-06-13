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

def partial_load(pretrained_dict, model):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)

def load_model():
    # Random seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if not opt.no_cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        
    model = models.CycleTime(class_num=params['n_classes'], 
                             trans_param_num=3, 
                             frame_gap=opt.frame_gap,
                             videoLen=opt.videoLen,
                             sample_duration=opt.sample_duration,
                             pretrained=opt.pretrained_imagenet, 
                             temporal_out=params['videoLen'], 
                             T=opt.T, 
                             hist=opt.hist,
                             batch_size=opt.batch_size)

    model = model.cuda()

    cudnn.benchmark = False
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters(), 
                lr=opt.learning_rate, 
                betas=(opt.momentum, 0.999), 
                weight_decay=opt.weight_decay)

    if opt.resume_path:
        # Load checkpoint.
        print('Loading checkpoint {}'.format(opt.resume_path))
        assert os.path.isfile(opt.resume_path), 'No checkpoint directory found'
        checkpoint = torch.load(opt.resume_path)
        # assert opt.arch == checkpoint['arch']
        opt.begin_epoch = checkpoint['epoch']

        partial_load(checkpoint['state_dict'], model)
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    return model

def test_and_eval(file, original_result_path, number, results_file_path): 

    model = load_model()

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
    
    val_json_name = 'temp'

    if number != 24:
        test.test(test_loader, model, opt, test_data.class_names, val_json_name)

    print("\n")
    print("EVALUATING")

    prediction_file = os.path.join(opt.result_path, 'val_{}.json'.format(val_json_name))
    general_output_path = os.path.join(opt.result_path, 'results_new_list.txt')
    subset = "validation"

    epoch, accuracy1, error1 = eval_hmdb51.eval_hmdb51(None, opt.annotation_path, prediction_file, subset, opt.top_k, number)
    epoch, accuracy5, error5 = eval_hmdb51.eval_hmdb51(None, opt.annotation_path, prediction_file, subset, 5, number)

    eval_results_1[epoch] = [accuracy5, error1]
    eval_results_5[epoch] = [accuracy5, error5]

    return eval_results_1, eval_results_5


if __name__ == '__main__':

    opt = parse_opts()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
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
    params, state = get_params(opt)

    
    folder = opt.result_path
    epoch1 = opt.begin_epoch

    eval_results_1 = {}
    eval_results_5 = {}


    for file in sorted(os.listdir(folder)):
        if file.endswith(".pth"):
            original_result_path = folder

            result_folder = folder.split("_hmdb51")[0]
            result_folder = result_folder
            
            path_file = os.path.join(folder, file)

            number = file.split("_")[1]
            number = int(number.split(".")[0])
            
            results_file = "results_{}.txt".format(number)
            results_file_path = os.path.join(original_result_path, results_file)
        
            if number >= int(epoch1):
                
                if os.path.isfile(results_file_path) is False:

                    print("Checkpoint:", file)
                    print("Resume_path:", opt.resume_path)
                    print("Result folder:", original_result_path)
                    print("Number:", number)
                    print("Results file path:", results_file_path)

                    opt.resume_path = os.path.join(original_result_path, file) 
                    print("Resume path:", opt.resume_path)

                    eval_results_1, eval_results_5 = test_and_eval(file, original_result_path, number, results_file_path)
                    
                    print("Eval 1 after ", number)
                    print(eval_results_1)
                    print("Eval 5 after ", number)
                    print(eval_results_5)

                    print("\n")
                    print("WRITING TO EVAL CHECKPOINT FILE")
                    print("\n")

                    file_1 = os.path.join(original_result_path, "eval_1_checkpoint_{}.txt".format(number))
                    file_2 = os.path.join(original_result_path, "eval_5_checkpoint_{}.txt".format(number))

                    print("File 1:", file_1)
                    print("File 2:", file_2)

                    fo_1 = open(file_1, 'w+')
                    for k, [v, w] in eval_results_1.items():
                        fo_1.write(str(k) + '\t'+ str(v) + '\t' + str(w) + '\n')
                    fo_1.close()

                    print("1 Done")

                    fo_2 = open(file_2, 'w+')
                    for k, [v, w] in eval_results_5.items():
                        fo_2.write(str(k) + '\t'+ str(v) + '\t' + str(w) + '\n')
                    fo_2.close()

                    print("5 Done")

            else:
                continue





