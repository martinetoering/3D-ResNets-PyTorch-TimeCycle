# TimeCycle

import random
from utils.imutils2 import *
from utils.transforms import *
import torchvision.transforms as transforms

import scipy.io as sio
import scipy.misc

# 3D-ResNets

import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import numpy as np

from dataset_utils import load_value_file


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        #print("This subset:", this_subset)
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 frame_gap, sample_duration, predDistance):
    print("\n") 

    data = load_annotation_data(annotation_path)

    print("Making dataset", subset)
    
    video_names, annotations = get_video_names_and_annotations(data, subset)

    # print("Video names", len(video_names))
    # print("Annotations", len(annotations))

    class_to_idx = get_class_labels(data)
    
    # print("Class_to_idx", len(class_to_idx))

    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    # print("Idx_to_class", len(idx_to_class))

    dataset = []
    gap_dataset = []
    path_to_id = {}
    name_to_sample = {}

    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('{} dataset loading [{}/{}]'.format(subset, i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1]
        }

        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]

        else:
            sample['label'] = -1

        path_to_id[video_path] = sample['label']

        name_to_sample[video_path] = sample

        gap_sample = copy.deepcopy(sample)


        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            
            gap_sample['frame_indices'] = list(range(1, n_frames + 1, frame_gap))
            
            dataset.append(sample)

            gap_dataset.append(gap_sample)

            # print("Sample:", sample)
        
            # exit() 

        else:
            if n_samples_for_each_video > 1:
                # print("Number of samples:", n_samples_for_each_video)
                # print("n_frames:", n_frames)
                # print("Sample duration:", sample_duration)

                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
                gap_step = max(1,
                           math.ceil((n_frames - 1 - sample_duration*frame_gap) /
                                     (n_samples_for_each_video - 1)))
                #print("step:", step)
            else:
                step = sample_duration

            #print("step:", step)

            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                gap_sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))

                gap_sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration*frame_gap), frame_gap))


                dataset.append(sample_j)

                gap_dataset.append(gap_sample_j)

                # print("Sample j:", sample_j)

            # exit() 

    return name_to_sample, dataset, gap_dataset, idx_to_class, path_to_id

def cropimg(img, offset_x, offset_y, cropsize):

    img = im_to_numpy(img)
    cropim = np.zeros([cropsize, cropsize, 3])
    cropim[:, :, :] = img[offset_y: offset_y + cropsize, offset_x: offset_x + cropsize, :]
    cropim = im_to_torch(cropim)

    return cropim

def cropimg_np(img, offset_x, offset_y, cropsize):

    cropim = np.zeros([cropsize, cropsize])
    cropim[:, :] = img[offset_y: offset_y + cropsize, offset_x: offset_x + cropsize]

    return cropim

def processflow(flow):
    boundnum = 60
    flow = flow.astype(np.float)
    flow = flow / 255.0
    flow = flow * boundnum * 2
    flow = flow - boundnum

    flow = np.abs(flow)

    return flow


class HMDB51(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 params,
                 root_path,
                 annotation_path,
                 subset,
                 frame_gap=1, 
                 sample_duration=13,
                 augment=['crop', 'flip', 'frame_gap'],
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 geometric_transform=None):

        if subset == "training":
            self.is_train = True
        else:
            self.is_train = False

        self.filelist = params['filelist']
        self.batch_size = params['batch_size']
        self.imgSize = params['imgSize']
        self.cropSize = params['cropSize']
        self.cropSize2 = params['cropSize2']
        self.videoLen = params['videoLen']

        # Prediction distance, how many frames far away
        self.predDistance = params['predDistance']

        # Offset x,y parameters
        self.offset = params['offset']
       
        # GridSize = 3
        self.gridSize = params['gridSize']

        self.frame_gap = frame_gap

        print("\n")
        print("Batch size:", self.batch_size)
        print("Videolen:", self.videoLen)
        print("Frame gap:", self.frame_gap)
        print("Prediction distance:", self.predDistance)
        print("Ofset:", self.offset)
        print("Grid size:", self.gridSize)

        self.augment = augment

        print("Augment:", self.augment)

        f = open(self.filelist, 'r')

        self.jpgfiles = []
        self.fnums = []

        for line in f:
            rows = line.split()
            jpgfile = rows[0]
            fnum = int(rows[1])

            if os.path.exists(jpgfile): 
                self.jpgfiles.append(jpgfile)
                self.fnums.append(fnum)

        f.close()

        # print("LENGTHL", len(self.jpgfiles))

        # exit()

        self.geometricTnf = geometric_transform

        print("\n")

        self.sample_duration = sample_duration
        self.name_to_sample, self.data, self.gap_data, self.class_names, self.path_to_id = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video, frame_gap,
            sample_duration, self.predDistance)

        print("Dataset:", subset)
        print("Number of samples for each video:", n_samples_for_each_video)
        print("Sample duration:", sample_duration)        

        if self.is_train:
            self.target_ids = []
            #print("JPG FILES:", len(self.jpgfiles))
            for path in self.jpgfiles:
                self.target_ids.append(self.path_to_id[path])
            #print("TARGETIDS:", len(self.target_ids))

        print("\n")
        print("Total Data", len(self.data))

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        self.backwards=False
        self.count = 0

        print("\n")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        if self.is_train:

            folder_path = self.jpgfiles[index]
            fnum = self.fnums[index]

            video = torch.Tensor(self.sample_duration, 3, self.cropSize, self.cropSize)
            imgs_target  = torch.Tensor(2, 3, self.cropSize, self.cropSize)
            patch_target = torch.Tensor(2, 3, self.cropSize, self.cropSize)

            future_imgs  = torch.Tensor(2, 3, self.cropSize, self.cropSize)

            # Random flip

            toflip = False
            if random.random() <= 0.5:
                toflip = True

            frame_gap = self.frame_gap
            current_len = (self.videoLen  + self.predDistance) * frame_gap
            sample_duration = self.sample_duration
            startframe = 0
            future_idx = current_len
            newLen = None

            if fnum >= (sample_duration + frame_gap):
                diff = fnum - (sample_duration + frame_gap)
                startframe = random.randint(0, diff)
                future_idx = startframe + current_len - 1

                # print("start frame: ", startframe)
                # print("future id:", future_idx)  

            else:
                newLen = int(fnum * 2.0 / 3.0)
                diffnum = fnum - newLen
                startframe = random.randint(0, diffnum)
                frame_gap = float(newLen - 1) / float(current_len)
                future_idx = int(startframe + current_len * frame_gap) - 1

                # print("Fnum:", fnum)
                # print("Newlen:", newLen)
                # print("start frame", startframe)
                # print("FUTURE ID:", future_idx)
                # print("folder_path:", folder_path)

            crop_offset_x = -1
            crop_offset_y = -1
            ratio = random.random() * (4/3 - 3/4) + 3/4
            
            video_indices = []

            if not newLen:
                for i in range(sample_duration):

                    nowid = int(startframe + i)
                    newid = nowid + 1

                    video_indices.append(newid)

                    newid = str(newid).zfill(5)
                    img_path = os.path.join(folder_path, "image_{}.jpg".format(newid))

                    img = load_image(img_path)  # CxHxW

                    ht, wd = img.size(1), img.size(2)

                    if ht <= wd:
                        ratio  = float(wd) / float(ht)
                        # width, height
                        img = resize(img, int(self.imgSize * ratio), self.imgSize)
                    else:

                        ratio  = float(ht) / float(wd)
                        # width, height
                        img = resize(img, self.imgSize, int(self.imgSize * ratio))


                    if crop_offset_x == -1:
                        crop_offset_x = random.randint(0, img.size(2) - self.cropSize - 1)
                        crop_offset_y = random.randint(0, img.size(1) - self.cropSize - 1)


                    img = cropimg(img, crop_offset_x, crop_offset_y, self.cropSize)

                    assert(img.size(1) == self.cropSize)
                    assert(img.size(2) == self.cropSize)

                    # Flip

                    if toflip:
                       img = torch.from_numpy(fliplr(img.numpy())).float()

                    mean=[0.485, 0.456, 0.406]
                    std=[0.229, 0.224, 0.225]
                    img = color_normalize(img, mean, std)


                    video[i] = img.clone()
                    
            else:
                for i in range(newLen):
                    
                    nowid = int(startframe + i)
                    newid = nowid + 1

                    video_indices.append(newid)


                    newid = str(newid).zfill(5)
                    img_path = os.path.join(folder_path, "image_{}.jpg".format(newid))

                    img = load_image(img_path)  # CxHxW

                    ht, wd = img.size(1), img.size(2)

                    if ht <= wd:
                        ratio  = float(wd) / float(ht)
                        # width, height
                        img = resize(img, int(self.imgSize * ratio), self.imgSize)
                    else:

                        ratio  = float(ht) / float(wd)
                        # width, height
                        img = resize(img, self.imgSize, int(self.imgSize * ratio))


                    if crop_offset_x == -1:
                        crop_offset_x = random.randint(0, img.size(2) - self.cropSize - 1)
                        crop_offset_y = random.randint(0, img.size(1) - self.cropSize - 1)


                    img = cropimg(img, crop_offset_x, crop_offset_y, self.cropSize)

                    assert(img.size(1) == self.cropSize)
                    assert(img.size(2) == self.cropSize)

                    # Flip

                    if toflip:
                       img = torch.from_numpy(fliplr(img.numpy())).float()

                    mean=[0.485, 0.456, 0.406]
                    std=[0.229, 0.224, 0.225]
                    img = color_normalize(img, mean, std)


                    video[i] = img.clone()


            future_imgs_indices = []

            for i in range(2):

                newid = int(future_idx + 1 + i * frame_gap)
                newid = newid + 1
                future_imgs_indices.append(newid)

                if newid > fnum:
                    newid = fnum

                newid = str(newid).zfill(5)
                img_path = os.path.join(folder_path, "image_{}.jpg".format(newid))

                img = load_image(img_path)  # CxHxW
                ht, wd = img.size(1), img.size(2)
                newh, neww = ht, wd
                if ht <= wd:
                    ratio  = float(wd) / float(ht)
                    # width, height
                    img = resize(img, int(self.imgSize * ratio), self.imgSize)
                    newh = self.imgSize
                    neww = int(self.imgSize * ratio)
                else:
                    ratio  = float(ht) / float(wd)
                    # width, height
                    img = resize(img, self.imgSize, int(self.imgSize * ratio))
                    newh = int(self.imgSize * ratio)
                    neww = self.imgSize

                img = cropimg(img, crop_offset_x, crop_offset_y, self.cropSize)
                assert(img.size(1) == self.cropSize)
                assert(img.size(2) == self.cropSize)

                if self.is_train:
                    if toflip:
                        img = torch.from_numpy(fliplr(img.numpy())).float()

                mean=[0.485, 0.456, 0.406]
                std=[0.229, 0.224, 0.225]
                img = color_normalize(img, mean, std)

                future_imgs[i] = img

                

            # print("FUtUrE IMGS:", future_imgs.size(), "VIDEO:", video.size())
            # print("Video indices are:", video_indices, "Of size:", len(video_indices), "Future imgs indices (first matters):", future_imgs_indices)

            for i in range(2):

                imgs_target[i] = future_imgs[i].clone()

            flow_cmb = future_imgs[0] - future_imgs[1]
            flow_cmb = im_to_numpy(flow_cmb)
            flow_cmb = flow_cmb.astype(np.float)
            flow_cmb = np.abs(flow_cmb)

            side_edge = self.cropSize
            box_edge  = int(side_edge / self.gridSize)

            lblxset = []
            lblyset = []
            scores  = []

            for i in range(self.gridSize - 2):
                for j in range(self.gridSize - 2):

                    offset_x1 = i * box_edge
                    offset_y1 = j * box_edge
                    lblxset.append(i)
                    lblyset.append(j)

                    tpatch = flow_cmb[offset_y1: offset_y1 + box_edge * 3, offset_x1: offset_x1 + box_edge * 3].copy()
                    tsum = tpatch.sum()
                    scores.append(tsum)


            scores = np.array(scores)
            ids = np.argsort(scores)
            ids = ids[-10: ]
            lbl = random.randint(0, 9)
            lbl = ids[lbl]

            lbl_x = lblxset[lbl]
            lbl_y = lblyset[lbl]


            if toflip:
                lbl_x = self.gridSize - 3 - lbl_x

            lbl   = lbl_x * (self.gridSize - 2) + lbl_y

            xloc = lbl_x / 6.0
            yloc = lbl_y / 6.0

            theta_aff = np.random.rand(6)
            scale = 1.0 - 1.0 / 3.0
            randnum = (np.random.rand(2) - 0.5) / 6.0
            xloc = xloc + randnum[0]
            yloc = yloc + randnum[1]

            if xloc < 0:
                xloc = 0.0
            if xloc > 1:
                xloc = 1.0

            if yloc < 0:
                yloc = 0.0
            if yloc > 1:
                yloc = 1.0

            # [-45, 45]
            alpha = (np.random.rand(1)-0.5)*2*np.pi*(1.0/4.0)

            theta_aff[2] = (xloc * 2.0 - 1.0) * scale
            theta_aff[5] = (yloc * 2.0 - 1.0) * scale
            theta_aff[0] = 1.0 / 3.0 *np.cos(alpha)
            theta_aff[1] = 1.0 / 3.0 *(-np.sin(alpha))
            theta_aff[3] = 1.0 / 3.0 *np.sin(alpha)
            theta_aff[4] = 1.0 / 3.0 *np.cos(alpha)

            theta = torch.Tensor(theta_aff.astype(np.float32))
            theta = theta.view(1, 2, 3)
            theta = theta.clone()
            theta_batch = theta.repeat(2, 1, 1)


            patch_target = self.geometricTnf(
                    image_batch=imgs_target, 
                    theta_batch=theta_batch)

            theta = theta.view(2, 3)
            imgs_target = imgs_target[0:1]
            patch_target = patch_target[0:1]

            sample = self.name_to_sample[folder_path]
            target = sample

            if self.target_transform is not None:
                target = self.target_transform(target)

            meta = {'folder_path': folder_path, 'startframe': startframe, 'future_idx': future_idx, 'frame_gap': float(frame_gap), 'crop_offset_x': crop_offset_x, 'crop_offset_y': crop_offset_y, 'dataset': 'vlog'}
            # print("Meta:", meta)

            return video, imgs_target, patch_target.data, theta, meta, target

        else:

            # toflip = False
            # if random.random() <= 0.5:
            #     toflip = True

            # crop_offset_x = -1
            # crop_offset_y = -1
            # ratio = random.random() * (4/3 - 3/4) + 3/4

            path = self.data[index]['video']

            frame_indices = self.data[index]['frame_indices']

            # print("Frame indices:", frame_indices)

            video = torch.Tensor(self.sample_duration, 3, self.cropSize, self.cropSize)

            for i, nowid in enumerate(frame_indices):
                
                newid = str(nowid).zfill(5)
                img_path = os.path.join(path, "image_{}.jpg".format(newid))

                img = load_image(img_path)  # CxHxW

                ht, wd = img.size(1), img.size(2)

                if ht <= wd:
                    ratio  = float(wd) / float(ht)
                    # width, height
                    img = resize(img, int(self.imgSize * ratio), self.imgSize)
                else:

                    ratio  = float(ht) / float(wd)
                    # width, height
                    img = resize(img, self.imgSize, int(self.imgSize * ratio))

                
                # Center crop, following KenshoHara

                width, height = img.size(2), img.size(1)

                x1 = int(round((width - 240) / 2.))
                y1 = int(round((height - 240) / 2.))

                img = cropimg(img, x1, y1, self.cropSize)

                assert(img.size(1) == self.cropSize)
                assert(img.size(2) == self.cropSize)

                # Flip

                # if toflip:
                #    img = torch.from_numpy(fliplr(img.numpy())).float()

                mean=[0.485, 0.456, 0.406]
                std=[0.229, 0.224, 0.225]
                img = color_normalize(img, mean, std)

                video[i] = img.clone()

            target = self.data[index]

            if self.target_transform is not None:
                target = self.target_transform(target)

            return video, target


    def __len__(self):
        return len(self.data)
