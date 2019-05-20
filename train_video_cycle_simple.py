'''
Training wiht VLOG
'''
from __future__ import print_function

import sys

# def info(type, value, tb):
#     if hasattr(sys, 'ps1') or not sys.stderr.isatty():
#     # we are in interactive mode or we don't have a tty-like
#     # device, so we call the default hook
#         sys.__excepthook__(type, value, tb)
#     else:
#         import traceback, pdb
#         # we are NOT in interactive mode, print the exception...
#         traceback.print_exception(type, value, tb)
#         print
#         # ...then start the debugger in post-mortem mode.
#         # pdb.pm() # deprecated
#         pdb.post_mortem(tb) # more "modern"

# sys.excepthook = info

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
from utils import Logger, AverageMeter, savefig

import models.dataset.vlog_train as vlog

from opts import parse_opts

global best_loss

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
    params['filelist'] = args.list
    params['imgSize'] = 256
    params['imgSize2'] = 320
    params['cropSize'] = 240
    params['cropSize2'] = 80
    params['offset'] = 0

    state = {k: v for k, v in args._get_kwargs()}

    params['predDistance'] = state['predDistance']
    print(params['predDistance'])

    params['batchSize'] = state['batchSize']
    print('batchSize: ' + str(params['batchSize']) )

    print('temperature: ' + str(state['T']))

    params['gridSize'] = state['gridSize']
    print('gridSize: ' + str(params['gridSize']) )

    params['classNum'] = state['classNum']
    print('classNum: ' + str(params['classNum']) )

    params['videoLen'] = state['videoLen']
    print('videoLen: ' + str(params['videoLen']) )

    return params, state

if __name__ == '__main__':
    opt = parse_opts()
    #print(torch.__version__)
    #exit()

    params, state = get_params(opt)
    args = opt

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    print(args.gpu_id)

    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

    best_loss = 0  # best test accuracy

    # Main

    print("Main")

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        os.mkdir(args.checkpoint)

    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        folder = opt.result_path
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    #print(opt)
    
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    model = models.CycleTime(class_num=params['classNum'], 
                             trans_param_num=3, 
                             pretrained=args.pretrained_imagenet, 
                             temporal_out=params['videoLen'], 
                             T=args.T, 
                             hist=args.hist)

    model_vc, parameters_vc = generate_model(opt)

    print("Model is made")

    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    cudnn.benchmark = False
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss().cuda()

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                    lr=args.u_lr, 
                    betas=(args.u_momentum, 0.999), 
                    weight_decay=args.u_wd)
    else:
        optimizer = optim.SGD(model.parameters(), 
                          lr=args.u_lr, 
                          weight_decay=args.u_wd, 
                          momentum=0.95
                          #dampening=0.9,
                          #nesterov=False
                          )

    print('weight_decay: ' + str(args.u_wd))
    print('beta1: ' + str(args.u_momentum))

    if len(args.pretrained) > 0:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.pretrained), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.pretrained)

        partial_load(checkpoint['state_dict'], model)
        # model.load_state_dict(checkpoint['state_dict'], strict=False)

        del checkpoint

    title = 'videonet'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.path_checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']

        partial_load(checkpoint['state_dict'], model)

        logger = Logger(os.path.join(args.path_checkpoint, 'log-resume.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Theta Loss', 'Theta Skip Loss'])

        logger_combined = Logger(os.path.join(args.path_checkpoint, 'log-resume-combined.txt'), title='combined')
        logger_combined.set_names(['Learning Rate', 'Combined loss', 'Acc'])
        del checkpoint

    else:
        logger = Logger(os.path.join(args.path_checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Theta Loss', 'Theta Skip Loss'])

        logger_combined = Logger(os.path.join(args.path_checkpoint, 'log-combined.txt'), title='combined')
        logger_combined.set_names(['Learning Rate', 'Combined loss', 'Acc'])

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    
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


    train_loader = torch.utils.data.DataLoader(
        vlog.VlogSet(params, 
                     opt.video_path,
                     opt.annotation_path,
                     'training',
                     is_train=True, 
                     frame_gap=args.frame_gap,
                     spatial_transform=spatial_transform,
                     temporal_transform=temporal_transform,
                     target_transform=target_transform),
        batch_size=params['batchSize'], 
        shuffle=True,
        num_workers=args.workers, 
        pin_memory=True)

    train_logger = Logger(
       os.path.join(opt.result_path, 'train.log'),
       ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(
       os.path.join(opt.result_path, 'train_batch.log'),
       ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    optimizer_vc = optim.SGD(
       parameters,
       lr=opt.learning_rate,
       momentum=opt.momentum,
       dampening=dampening,
       weight_decay=opt.weight_decay,
       nesterov=opt.nesterov)
    scheduler = lr_scheduler.ReduceLROnPlateau(
       optimizer_vc, 
       'min', 
       patience=opt.lr_patience)

    print(train_loader)

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Train and val
    for epoch in range(start_epoch, args.n_epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['u_lr']))

        loss, acc_vc, train_loss, theta_loss, theta_skip_loss = train(params, 
            train_loader, model, criterion, optimizer, epoch, use_cuda, args)

        # append logger file
        # print("TRAIN LOSS:", train_loss[0])
        # print("THETA LOSS:", theta_loss[0])
        # print("THETA_SKIP_LOSS", theta_skip_loss[0])
        logger.append([state['u_lr'], train_loss[0], theta_loss[0], theta_skip_loss[0]])

        if epoch % opt.checkpoint == 0:
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, checkpoint=args.path_checkpoint)

        # append logger file
        print("LOSS:", loss[0])
        logger_combined.append([state['u_lr'], state['learning_rate'], loss[0], acc_vc[0]])

        # if epoch % opt.checkpoint == 0:
        #     save_checkpoint({
        #             'epoch': epoch + 1,
        #             'state_dict': model.state_dict(),
        #             'optimizer': optimizer.state_dict()
        #         }, checkpoint=args.path_checkpoint)

    logger.close()

    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = VideoID()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt, test_data.class_names)

    if opt.eval:
        name = opt.result_path + '/' + folder + '_' + str(opt.n_epochs) + '.txt'
        print("Name:", name)
        if opt.test_subset == "val":
            prediction = os.path.join(opt.result_path, "val.json")
            subset = "validation"
        if opt.test_subset == "test":
            prediction = os.path.join(opt.result_path, "test.json")
            subset = "testing"
        eval_hmdb51.eval_hmdb51(name, opt.annotation_path, prediction, subset, 1)


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()

def train(params, train_loader, model, criterion, optimizer, epoch, 
          use_cuda, args, model_vc, optimizer_vc, opt, epoch_logger, 
          batch_logger):
    # switch to train mode
    model.train()
    # model.apply(set_bn_eval)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    main_loss = AverageMeter()
    losses_theta = AverageMeter()
    losses_theta_skip = AverageMeter()

    losses_dict = dict(
        cnt_trackers=None,
        back_inliers=None,
        loss_targ_theta=None,
        loss_targ_theta_skip=None
    )

    end = time.time()

    batch_time_vc = AverageMeter()
    data_time_vc = AverageMeter()
    losses_vc = AverageMeter()
    accuracies_vc = AverageMeter()

    end_time_vc = time.time()

    print("Training starts")


    for batch_idx, (imgs, img, patch2, theta, meta, clip, target) in enumerate(train_loader):
         

        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        # optimizerC.zero_grad()

        if imgs.size(0) < params['batchSize']:
            break

        imgs = torch.autograd.Variable(imgs.cuda())
        img = torch.autograd.Variable(img.cuda())
        patch2 = torch.autograd.Variable(patch2.cuda())
        theta = torch.autograd.Variable(theta.cuda())

        folder_paths = meta['folder_path']
        startframes = meta['startframe']
        future_idxs = meta['future_idx']

        outputs = model(imgs, patch2, img, theta)

        losses = model.loss(*outputs)
        loss_targ_theta, loss_targ_theta_skip, loss_back_inliers = losses

        loss = sum(loss_targ_theta) / len(loss_targ_theta) * args.lamda + \
            sum(loss_back_inliers) / len(loss_back_inliers) + \
            loss_targ_theta_skip[0] * args.lamda

        outstr = ''

        main_loss.update(loss_back_inliers[0].data, imgs.size(0))
        outstr += '| Loss: %.3f' % (main_loss.avg)

        losses_theta.update(sum(loss_targ_theta).data / len(loss_targ_theta), imgs.size(0))
        losses_theta_skip.update(sum(loss_targ_theta_skip).data / len(loss_targ_theta_skip), imgs.size(0))

        def add_loss_to_str(name, _loss):
            outstr = ' | %s '% name
            if losses_dict[name] is None:
                losses_dict[name] = [AverageMeter() for _ in _loss]

            for i,l in enumerate(_loss):
                losses_dict[name][i].update(l.data, imgs.size(0))
                outstr += ' %s: %.3f ' % (i, losses_dict[name][i].avg)
            return outstr

        outstr += add_loss_to_str('loss_targ_theta', loss_targ_theta)
        outstr += add_loss_to_str('loss_targ_theta_skip', loss_targ_theta_skip)

        # 3D-ResNets

        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs_vc = model(inputs)
        loss_vc = criterion(outputs_vc, targets)
        acc_vc = calculate_accuracy(outputs_vc, targets)

        losses_vc.update(loss_vc.data[0], inputs.size(0))
        accuracies_vc.update(acc_vc, inputs.size(0))

        loss = loss + loss_vc

        combined_str += add_loss_to_str('loss_combined', loss)

        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 5 == 0:
            outstr  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | {outstr}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    outstr=outstr
                    )
            print(outstr)

        batch_time_vc.update(time.time() - end_time)
        end_time_vc = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(train_loader) + (i + 1),
            'loss': losses_vc.val,
            'acc': accuracies_vc.val,
            'lr': optimizer_vc.param_groups[0]['lr']
        })

        # print('Epoch: [{0}][{1}/{2}]\t'
        #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #       'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
        #           epoch_vc,
        #           i + 1,
        #           len(train_loader),
        #           batch_time=batch_time_vc,
        #           data_time=data_time_vc,
        #           loss=losses_vc,
        #           acc=accuracies_vc))

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(train_loader),
                  batch_time=batch_time_vc,
                  data_time=data_time_vc,
                  loss=loss,
                  acc=accuracies_vc))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses_vc.avg,
        'acc': accuracies_vc.avg,
        'lr': optimizer_vc.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model_vc.state_dict(),
            'optimizer': optimizer_vc.state_dict(),
        }
        torch.save(states, save_file_path)



    return loss.avg, acc_vc.avg, main_loss.avg, losses_theta.avg, losses_theta_skip.avg

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

if __name__ == '__main__':

    args = parse_opts()
    train_video_cycle(args)
