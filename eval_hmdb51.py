import json
import os
import sys

import numpy as np
import pandas as pd

def eval_hmdb51(name, annotation_path, prediction_path, test_subset, top_k, name_general, epoch):
    hmdb_classification = HMDBclassification(name, annotation_path, prediction_path, subset=test_subset, top_k=top_k)

    hmdb_classification.evaluate(name, print_per_epoch)

    if name_general:
        f = open(name_general, "a")
        f.write(str(epoch) + '          ' + str(hmdb_classification.hit_at_k) + '\n') 
        f.close()

class HMDBclassification(object):

    def __init__(self, name, ground_truth_filename=None, prediction_filename=None,
                 subset='validation', verbose=False, top_k=1):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.verbose = verbose
        self.top_k = top_k
        self.ap = None
        self.hit_at_k = None
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        
        with open(name, 'w') as f:
            f.write('{}\n'.format(name))
            print("Write evaluation results to file:", name)
            f.write('[INIT] Loaded annotations from {} subset.\n'.format(subset))
            nr_gt = len(self.ground_truth)
            f.write('\tNumber of ground truth instances: {}\n'.format(nr_gt))
            nr_pred = len(self.prediction)
            f.write('\tNumber of predictions: {}\n'.format(nr_pred))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        # if not all([field in data.keys() for field in self.gt_fields]):
            # raise IOError('Please input a valid ground truth file.')

        # Initialize data frame
        activity_index, cidx = {}, 0
        video_lst, label_lst = [], []
        for videoid, v in data['database'].items():
            if self.subset != v['subset']:
                continue
            this_label = v['annotations']['label']
            if this_label not in activity_index:
                activity_index[this_label] = cidx
                cidx += 1
            video_lst.append(videoid)
            label_lst.append(activity_index[this_label])
        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     'label': label_lst})
        ground_truth = ground_truth.drop_duplicates().reset_index(drop=True)
        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        # if not all([field in data.keys() for field in self.pred_fields]):
            # raise IOError('Please input a valid prediction file.')

        # Initialize data frame
        video_lst, label_lst, score_lst = [], [], []
        for videoid, v in data['results'].items():
            for result in v:
                label = self.activity_index[result['label']]
                video_lst.append(videoid)
                label_lst.append(label)
                score_lst.append(result['score'])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

    def evaluate(self, name, print_per_epoch):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        hit_at_k = compute_video_hit_at_k(self.ground_truth,
                                          self.prediction, top_k=self.top_k)
        with open(name, 'w') as f:
            f.write('[RESULTS] Performance on ActivityNet untrimmed video '
               'classification task.\n')
            f.write('\tError@{}: {}\n'.format(self.top_k, 1.0 - hit_at_k))
            #print '\tAvg Hit@{}: {}'.format(self.top_k, avg_hit_at_k)
        self.hit_at_k = hit_at_k

################################################################################
# Metrics
################################################################################
def compute_video_hit_at_k(ground_truth, prediction, top_k=3):
    """Compute accuracy at k prediction between ground truth and
    predictions data frames. This code is greatly inspired by evaluation
    performed in Karpathy et al. CVPR14.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 'label']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 'label', 'score']

    Outputs
    -------
    acc : float
        Top k accuracy score.
    """
    video_ids = np.unique(ground_truth['video-id'].values)
    avg_hits_per_vid = np.zeros(video_ids.size)
    for i, vid in enumerate(video_ids):
        pred_idx = prediction['video-id'] == vid
        if not pred_idx.any():
            continue
        this_pred = prediction.loc[pred_idx].reset_index(drop=True)
        # Get top K predictions sorted by decreasing score.
        sort_idx = this_pred['score'].values.argsort()[::-1][:top_k]
        this_pred = this_pred.loc[sort_idx].reset_index(drop=True)
        # Get labels and compare against ground truth.
        pred_label = this_pred['label'].tolist()
        gt_idx = ground_truth['video-id'] == vid
        gt_label = ground_truth.loc[gt_idx]['label'].tolist()
        avg_hits_per_vid[i] = np.mean([1 if this_label in pred_label else 0
                                       for this_label in gt_label])
    return float(avg_hits_per_vid.mean())


if __name__ == '__main__':
    name = sys.argv[1]
    ground_truth_path = sys.argv[2]
    prediction_path = sys.argv[3]
    test_subset = sys.argv[4]
    name_general = sys.argv[5]
    epoch = sys.argv[6]
    eval_hmdb51(name, ground_truth_path, prediction_path, test_subset=test_subset, top_k=1, name_general=name_general, epoch=epoch)
