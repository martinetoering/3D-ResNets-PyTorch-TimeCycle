import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import json

from dataset_utils import AverageMeter


from json import JSONEncoder
class MyEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__    


def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=10)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i]],
            'score': sorted_scores[i]
        })

    test_results['results'][video_id] = video_results


def test(data_loader, model, opt, class_names, val_json_name):
    print('test')
    print(torch.__version__)
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}}
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        inputs = Variable(inputs.cuda(), volatile=True)
        _, _, _, outputs, _, _ = model.forward_base(inputs)

        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs)

        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and targets[j] != previous_video_id:
                calculate_video_results(output_buffer, previous_video_id,
                                        test_results, class_names)

                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[j]
        

        if (i % 100) == 0:
            with open(
                    os.path.join(opt.result_path, '{}_{}.json'.format(
                        opt.test_subset, val_json_name)), 'w') as f:
                MyEncoder().encode(f)
                json.dump(test_results, f, cls=MyEncoder)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time))

    print("Test results:", test_results)

    with open(
            os.path.join(opt.result_path, '{}_{}.json'.format(opt.test_subset, val_json_name)),
            'w') as f:
        json.dump(test_results, f)
