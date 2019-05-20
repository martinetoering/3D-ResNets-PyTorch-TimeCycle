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

    losses = AverageMeter()
    accuracies = AverageMeter()

    main_loss = AverageMeter()
    losses_vc = AverageMeter()
    losses_combined = AverageMeter()
    losses_theta = AverageMeter()
    losses_theta_skip = AverageMeter()

    losses_dict = dict(
        cnt_trackers=None,
        back_inliers=None,
        loss_targ_theta=None,
        loss_targ_theta_skip=None
    )

    end_time = time.time()
    
    for i, (imgs, img, patch2, theta, meta, targets) in enumerate(data_loader):
        
        # Measure data loading time
        data_time.update(time.time() - end_time)

        if imgs.size(0) < params['batch_size']:
            break

        imgs = Variable(imgs.cuda())
        img = Variable(img.cuda())
        patch2 = Variable(patch2.cuda())
        theta = Variable(theta.cuda())

        folder_paths = meta['folder_path']
        startframes = meta['startframe']
        future_idxs = meta['future_idx']

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        targets = Variable(targets)

        outputs_vc, outputs = model(imgs, patch2, img, theta)

        loss_vc = criterion(outputs_vc, targets)
        acc_vc = calculate_accuracy(outputs_vc, targets)


        losses = model.loss(*outputs)
        loss_targ_theta, loss_targ_theta_skip, loss_back_inliers = losses

        loss = sum(loss_targ_theta) / len(loss_targ_theta) * opt.lamda + \
            sum(loss_back_inliers) / len(loss_back_inliers) + \
            loss_targ_theta_skip[0] * opt.lamda

        main_loss.update(loss_back_inliers[0].data, imgs.size(0))

        losses_theta.update(sum(loss_targ_theta).data / len(loss_targ_theta), imgs.size(0))
        losses_theta_skip.update(sum(loss_targ_theta_skip).data / len(loss_targ_theta_skip), imgs.size(0))

        
        losses_vc.update(loss_vc.data[0], imgs.size(0))

        losses_combined.update(loss_vc.data[0] + loss_back_inliers[0].data, imgs.size(0))

        accuracies.update(acc_vc, imgs.size(0))
       
        optimizer.zero_grad()        
        # Combine 
        loss = loss + loss_vc
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)

        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # print("Loss:", type(losses_combined.val),  "Loss_vc:", type(losses_vc.val), "Loss_main", type(main_loss.val))

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': (losses_combined.val)[0],
            'loss_vc': losses_vc.val,
            'loss_main': (main_loss.val)[0],
            'acc': accuracies.val,
            'lr': get_lr(optimizer)
        })

        
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  acc=accuracies))

        # print("Epoch:", epoch, "[", i+1, "/", len(data_loader), "]")
        # print('Epoch: [{0}][{1}/{2}]\t'
        #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #       'Loss {losses_combined.val:.4f} ({losses_combined.avg:.4f})\t'
        #       'Loss_vc {losses_vc.val:.4f} ({losses_vc.avg:.4f})\t'
        #       'Loss_main {main_loss.val:.4f} ({main_loss.avg:.4f})\t'
        #       'Loss_theta {losses_theta.val:.4f} ({losses_theta.avg:.4f})\t'
        #       'Loss_theta_skip {losses_theta_skip.val:.4f} ({losses_theta_skip.avg:.4f})\t'
        #       'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
        #           epoch,
        #           i + 1,
        #           len(data_loader),
        #           batch_time=batch_time,
        #           data_time=data_time,
        #           loss=losses_combined,
        #           loss_vc=losses_vc,
        #           loss_main=main_loss,
        #           loss_theta=losses_theta,
        #           loss_theta_skip=losses_theta_skip,
        #           acc=accuracies))

    epoch_logger.log({
        'epoch': epoch,
        'loss': (losses_combined.avg)[0],
        'loss_vc': losses_vc.avg,
        'loss_main': (main_loss.avg)[0],
        'acc': accuracies.avg,
        'lr': get_lr(optimizer)
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

    return losses_vc.avg, accuracies, main_loss.avg, losses_theta.avg, losses_theta_skip.avg
