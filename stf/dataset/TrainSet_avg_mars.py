from .Dataset import Dataset
from ..utils.dataset_utils import parse_im_name

import os.path as osp
from PIL import Image
import cv2
import numpy as np
import random


class TrainSetAvgMARS(Dataset):
    """Training set for identification loss.
    Args:
      ids2labels: a dict mapping ids to labels
    """

    def __init__(self,
                 im_dir=None,
                 im_names=None,
                 ids2labels=None,
                 class_balance=False,
                 camera_weight=False,
                 images_per_id=8,
                 max_n_samples=25,
                 p_avg=0.5,
                 **kwargs):
        # The im dir of all images
        self.im_dir = im_dir
        self.im_names = im_names
        self.ids2labels = ids2labels
        self.im_dict = {}
        self.class_balance = class_balance
        self.camera_weight = camera_weight
        self.images_per_id = images_per_id
        self.max_n_samples = max_n_samples
        for im_name in self.im_names:
            id_ch = '_'.join(im_name.split('_')[0:2])
            if id_ch not in self.im_dict:
                self.im_dict[id_ch] = []
            self.im_dict[id_ch].append(im_name)

        dataset_size = len(self.im_dict.keys())
        self.epoch_done = True
        self.id_list = list(self.im_dict.keys())
        print('Using TrainSetAvgMARS')
        super(TrainSetAvgMARS, self).__init__(
            dataset_size=dataset_size, **kwargs)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, key):
        return self.get_sample(key)

    def get_sample(self, ptr):
        """Get one sample to put to queue."""
        if ptr >= len(self.id_list):
            ptr = ptr % len(self.id_list)
        im_names = []
        #print (len(self.id_list), ptr)
        id = self.id_list[ptr]
        if random.random() < 0:
            n_sample = 1
        else:
            n_sample = random.randint(min(5, len(self.im_dict[self.id_list[ptr]]), self.max_n_samples), min(
                self.max_n_samples, len(self.im_dict[self.id_list[ptr]])))

        im_names = random.sample(self.im_dict[id], n_sample)
        # print im_names, len(im_names)
        ims = np.zeros(
            (self.max_n_samples, 3, self.pre_process_im.resize_h_w[0], self.pre_process_im.resize_h_w[1]))
#    print id, 'foo'
        for i, im_name in enumerate(im_names):
            im_path = osp.join(self.im_dir, im_name)
            #im = np.asarray(Image.open(im_path))
#      print im_name
            im = cv2.imread(im_path)
            if im is None:
                while im is None:
                    im_name = random.choice(self.im_dict[self.id_list[ptr]])
                    im_path = osp.join(self.im_dir, im_name)
                    im = cv2.imread(im_path)
            im = im[:, :, ::-1]
            # print np.sum(im-im0), 'foo'
            im, mirrored = self.pre_process_im(im)
            ims[i] = im
        label = self.ids2labels[int(id.split('_')[0])]
        sample_mask = np.array([1] * n_sample + [0] *
                               (self.max_n_samples - n_sample))
        return (ims, im_names, label, mirrored, sample_mask)

    def next_batch(self):
        """Next batch of images and labels.
        Returns:
          ims: numpy array with shape [N, H, W, C] or [N, C, H, W], N >= 1
          im_names: a numpy array of image names, len(im_names) >= 1
          labels: a numpy array of image labels, len(labels) >= 1
          mirrored: a numpy array of booleans, whether the images are mirrored
          self.epoch_done: whether the epoch is over
        """
        if self.epoch_done and self.shuffle:
            #      if self.class_balance or self.camera_weight:
            self.prng.shuffle(self.id_list)
#      else:
#        self.prng.shuffle(self.im_names)

#    if self.class_balance:
#    	samples, self.epoch_done = self.prefetcher.next_batch(self.im_dict,self.ids2labels,self.im_dir)
#    else:
        samples, self.epoch_done = self.prefetcher.next_batch_test()
        im_list, im_names, labels, mirrored, sample_mask = zip(*samples)
        # print labels
        # print im_names
        # print len(im_names)
        # Transform the list into a numpy array with shape [N, ...]
        ims = np.stack(im_list, axis=0)
        im_names = np.array(im_names)
        labels = np.array(labels)
        mirrored = np.array(mirrored)
        sample_mask = np.array(sample_mask)
        return ims, im_names, labels, mirrored, self.epoch_done, sample_mask
