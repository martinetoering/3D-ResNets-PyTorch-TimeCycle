import torch
from torch.autograd import Variable
import time
import torch.nn.functional as F
import sys
import os
import json

from dataset_utils import AverageMeter, calculate_accuracy


# from json import JSONEncoder
# class MyEncoder(JSONEncoder):
#         def default(self, o):
#             return o.__dict__    


# def calculate_video_results(output_buffer, video_id, test_results, class_names):
#     video_outputs = torch.stack(output_buffer)
#     average_scores = torch.mean(video_outputs, dim=0)
#     sorted_scores, locs = torch.topk(average_scores, k=10)

#     video_results = []
#     for i in range(sorted_scores.size(0)):
#         video_results.append({
#             'label': class_names[locs[i]],
#             'score': sorted_scores[i]
#         })

#     test_results['results'][video_id] = video_results

# def val_test_eval_epoch(epoch, params, data_loader, model, criterion, opt, logger, class_names):
#     print('Validation at epoch {}'.format(epoch))
#     print('test')
#     print(torch.__version__)

#     model.eval()

#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     accuracies = AverageMeter()

#     end_time = time.time()
#     output_buffer = []
#     previous_video_id = ''
#     test_results = {'results': {}}
#     for i, (inputs, targets) in enumerate(data_loader):

#         data_time.update(time.time() - end_time)


#         #if not opt.no_cuda:
#         #    targets = targets.cuda(async=True)
#         inputs = Variable(inputs.cuda(), volatile=True)

#         _, _, _, outputs = model.forward_base(inputs)

#         # loss = criterion(outputs, targets)
#         # acc = calculate_accuracy(outputs, targets)

#         if not opt.no_softmax_in_test:
#             outputs = F.softmax(outputs)

#         # losses.update(loss.data[0], inputs.size(0))
#         # accuracies.update(acc, inputs.size(0))


#         for j in range(outputs.size(0)):
#             if not (i == 0 and j == 0) and targets[j] != previous_video_id:
#                 calculate_video_results(output_buffer, previous_video_id,
#                                         test_results, class_names)

#                 output_buffer = []
#             output_buffer.append(outputs[j].data.cpu())
#             previous_video_id = targets[j]
        

#         if (i % 100) == 0:
#             with open(
#                     os.path.join(opt.result_path, '{}_{}.json'.format(
#                         opt.test_subset, epoch)), 'w') as f:
#                 MyEncoder().encode(f)
#                 json.dump(test_results, f, cls=MyEncoder)


#         batch_time.update(time.time() - end_time)
#         end_time = time.time()

#         print('Epoch: [{0}][{1}/{2}]\t'
#               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#               'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#               'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
#                   epoch,
#                   i + 1,
#                   len(data_loader),
#                   batch_time=batch_time,
#                   data_time=data_time,
#                   loss=losses,
#                   acc=accuracies))

#     # print("Test results:", test_results)

#     file_json = os.path.join(opt.result_path, '{}_{}.json'.format(opt.test_subset, epoch))
#     with open(
#             file_json,
#             'w') as f:
#         json.dump(test_results, f)

#     logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

#     return losses.avg, file_json


def val_epoch(epoch, params, data_loader, model, criterion, opt, logger):
    print('Validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):

        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs.cuda(), volatile=True)
        targets = Variable(targets, volatile=True)
        _, _, _, outputs = model.forward_base(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.data[0], inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    return losses.avg


    
