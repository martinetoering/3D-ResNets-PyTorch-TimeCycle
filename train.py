import torch
from torch.autograd import Variable
import time
import os
import sys

from dataset_utils import AverageMeter, calculate_accuracy

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_epoch(epoch, params, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    
    # Switch to train mode

    print('Train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
  
    # TimeCycle

    main_loss = AverageMeter() # The feature similarity
    losses_theta = AverageMeter()
    losses_theta_skip = AverageMeter() 
    losses_overall = AverageMeter() # Combination of the three


    losses_dict = dict(
        cnt_trackers=None,
        back_inliers=None,
        loss_targ_theta=None,
        loss_targ_theta_skip=None
    )
    
    # Binary classification

    losses_bin = AverageMeter()
    accs_bin = AverageMeter()

    # HMDB Classification

    losses_vc = AverageMeter()
    accuracies = AverageMeter()

    # Combined

    losses_combined = AverageMeter()


    end_time = time.time()
    
    for i, (video, img, patch2, theta, meta, targets, targets_bin) in enumerate(data_loader):

        # Measure data loading time

        data_time.update(time.time() - end_time)

        if video.size(0) < params['batch_size']:
            break

        video = Variable(video.cuda())
        img = Variable(img.cuda())
        patch2 = Variable(patch2.cuda())
        theta = Variable(theta.cuda())

        folder_paths = meta['folder_path']
        startframes = meta['startframe']
        future_idxs = meta['future_idx']

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
            targets_bin = targets_bin.cuda(async=True)
        
        targets = Variable(targets)
        targets_bin = Variable(targets_bin)

        outputs_bin, outputs_vc, outputs = model(video, patch2, img, theta)

        # HMDB video classification

        loss_vc = criterion(outputs_vc, targets)
        acc_vc = calculate_accuracy(outputs_vc, targets)

        losses_vc.update(loss_vc.data[0], video.size(0))
        accuracies.update(acc_vc, video.size(0))

        # Binary classification 

        loss_bin = criterion(outputs_bin, targets_bin)
        acc_bin = calculate_accuracy(outputs_bin, targets_bin)

        accs_bin.update(acc_bin, video.size(0))

        # TimeCycle

        losses = model.loss(*outputs)
        loss_targ_theta, loss_targ_theta_skip, loss_back_inliers = losses

        loss = sum(loss_targ_theta) / len(loss_targ_theta) * opt.lamda + \
            sum(loss_back_inliers) / len(loss_back_inliers) + \
            loss_targ_theta_skip[0] * opt.lamda

        main_loss.update(loss_back_inliers[0].data, video.size(0))
        losses_theta.update(sum(loss_targ_theta).data / len(loss_targ_theta), video.size(0))
        losses_theta_skip.update(sum(loss_targ_theta_skip).data / len(loss_targ_theta_skip), video.size(0))

        # Apply weights

        loss = opt.timecycle_weight*loss
        losses_overall.update(loss[0].data, video.size(0))

        loss_bin = opt.binary_class_weight*loss_bin
        losses_bin.update(loss_bin.data[0], video.size(0))

        # Combine losses

        loss_combined = loss + loss_vc + loss_bin

        losses_combined.update(loss_combined[0].data, video.size(0))
       
        optimizer.zero_grad()        
        
        # Combine losses

        loss_combined.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)

        optimizer.step()

        # Measure elapsed time
        
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': (losses_combined.val)[0],
            'loss_hmdb_class': losses_vc.val,
            'loss_timecycle': (losses_overall.val)[0],
            'loss_bin_class': losses_bin.val,
            'acc': accuracies.val,
            'acc_bin': accs_bin.val,
            'lr': get_lr(optimizer),
            'loss_sim': (main_loss.val)[0],
            'theta_loss': (losses_theta.val)[0],
            'theta_skip_loss': (losses_theta_skip.val)[0]
        })

        
        print('Epoch: [{0}][{1}/{2}] '
              'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) '
              'Data {data_time.val:.2f} ({data_time.avg:.2f})\t'
              'Loss {losses_combined.val[0]:.3f} ({losses_combined.avg[0]:.3f})\t'
              'L_hmdb {losses_vc.val:.3f} ({losses_vc.avg:.3f})\t'
              'L_time {losses_overall.val[0]:.3f} ({losses_overall.avg[0]:.3f})\t'
              'L_bin {losses_bin.val:.3f} ({losses_bin.avg:.3f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
              'Acc_bin {accs_bin.val:.3f} ({accs_bin.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  losses_combined=losses_combined,
                  losses_vc=losses_vc,
                  losses_bin=losses_bin,
                  losses_overall=losses_overall,
                  acc=accuracies,
                  accs_bin=accs_bin))


    epoch_logger.log({
        'epoch': epoch,
        'loss': (losses_combined.avg)[0],
        'loss_hmdb_class': losses_vc.avg,
        'loss_timecycle': (losses_overall.avg)[0],
        'loss_bin_class': losses_bin.avg,
        'acc': accuracies.avg,
        'acc_bin': accs_bin.avg,
        'lr': get_lr(optimizer),
        'loss_sim': (main_loss.avg)[0],
        'theta_loss': (losses_theta.avg)[0],
        'theta_skip_loss': (losses_theta_skip.avg)[0]
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)

