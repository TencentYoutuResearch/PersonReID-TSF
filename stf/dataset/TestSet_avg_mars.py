from __future__ import print_function
import sys
import time
import os.path as osp
from PIL import Image
import cv2
import numpy as np
import random
from collections import defaultdict

from .Dataset import Dataset

from ..utils.utils import measure_time
from ..utils.re_ranking import re_ranking
from ..utils.metric import cmc, mean_ap, precision_recall, evaluate
from ..utils.dataset_utils import parse_im_name
from ..utils.distance import normalize
from ..utils.distance import compute_dist

import pickle

DEBUG = True


class TestSetAvgMARS(Dataset):
    """
    Args:
      extract_feat_func: a function to extract features. It takes a batch of
        images and returns a batch of features.
      marks: a list, each element e denoting whether the image is from 
        query (e == 0), or
        gallery (e == 1), or 
        multi query (e == 2) set
    """

    def __init__(
            self,
            im_dir=None,
            im_names=None,
            marks=None,
            extract_feat_func=None,
            separate_camera_set=None,
            single_gallery_shot=None,
            first_match_break=None,
            **kwargs):

        # The im dir of all images
        self.im_dir = im_dir
        self.im_names = im_names
        self.extract_feat_func = extract_feat_func
        self.separate_camera_set = separate_camera_set
        self.single_gallery_shot = single_gallery_shot
        self.first_match_break = first_match_break
        self.im_dict = {}
        self.marks = {}

        self.max_n_samples = 25
        '''
    self.im_names = self.im_names[0:1000]
    self.im_names += im_names[-5000:]
    marks = marks[0:1000] + marks[-5000:]
    '''

        id_ch_segment = {}

        # self.im_names.sort()
        #self.im_names = self.im_names[0:250] + self.im_names[10000:10250] + self.im_names[-250:]
        # marks = [0] * 250 + [1] * 500#list(marks[0:250] + marks[1000:1250] + marks[-250:])
        for i, im_name in enumerate(self.im_names):
            id_ch = '_'.join(im_name.split('_')[0:2])
            if id_ch not in id_ch_segment:
                id_ch_segment[id_ch] = [id_ch + '_seg00000']
                self.im_dict[id_ch + '_seg00000'] = []
                self.marks[id_ch + '_seg00000'] = []

            key = id_ch_segment[id_ch][-1]

            if len(self.im_dict[key]) == self.max_n_samples:
                key = id_ch + \
                    '_seg%05d' % (
                        int(key.split('_')[-1].replace('seg', '')) + 1)
                id_ch_segment[id_ch].append(key)
                self.im_dict[key] = []
                self.marks[key] = []

            self.im_dict[key].append(im_name)
            self.marks[key].append(marks[i])

        id_list = sorted(list(self.im_dict.keys()))
        self.id_list = id_list
        self.id_ch_segment = id_ch_segment
        super(TestSetAvgMARS, self).__init__(
            dataset_size=len(self.id_list), **kwargs)
        print('Creating dataset using TestSetAvgMARS')

    def set_feat_func(self, extract_feat_func):
        self.extract_feat_func = extract_feat_func

    def get_sample(self, ptr):
        """get one id in one cam's images to queue"""
        if ptr >= len(self.id_list):
            ptr = ptr % len(self.id_list)
        im_names = []
        id_ch = self.id_list[ptr]
        im_names = self.im_dict[id_ch]
        # if len(im_names) > self.max_n_samples:
        #  indices = random.sample(range(len(im_names)), self.max_n_samples)
        #  im_names = [im_names[i] for i in indices]
        #print (len(im_names))
        ims = np.zeros(
            (self.max_n_samples, 3, self.pre_process_im.resize_h_w[0], self.pre_process_im.resize_h_w[1]))
        for i, im_name in enumerate(im_names):
            im_path = osp.join(self.im_dir, im_name)
            im = cv2.imread(im_path)

            if im is None:
                print('%s img read fail' % im_path)
                continue
            im = im[:, :, ::-1]
            im, _ = self.pre_process_im(im)
            ims[i] = np.copy(im)
        id = id_ch
        cam = id_ch.split('_')[1][0]
        track = id_ch.split('_')[1][1:]

        mark = self.marks[id_ch][0]

        sample_mask = np.array([1] * len(im_names) + [0]
                               * (self.max_n_samples - len(im_names)))

        return (ims, im_names, id, cam, track, sample_mask, mark)

    def next_batch(self):
        if self.epoch_done and self.shuffle:
            self.prng.shuffle(self.im_names)
        ims = None
        im_names = None
        ids = None
        cams = None
        tracks = None
        sample_masks = None
        marks = None
        samples, self.epoch_done = self.prefetcher.next_batch_test()
        if len(samples) > 0:
            ims_list, im_names_list, ids, cams, tracks, sample_masks, marks = zip(
                *samples)
        else:
            return ims, im_names, ids, cams, tracks, sample_masks, marks, self.epoch_done
        # Transform the list into a numpy array with shape [N, ...]
        ims = np.stack(ims_list, axis=0)
        ids = np.array(ids)
        cams = np.array(cams)
        tracks = np.array(tracks)
        im_names = im_names_list
        sample_masks = np.array(sample_masks)
        marks = np.array(marks)
        return ims, im_names, ids, cams, tracks, sample_masks, marks, self.epoch_done
    def extract_feat(self, normalize_feat, verbose=True):
        """Extract the features of the whole image set.
        Args:
          normalize_feat: True or False, whether to normalize feature to unit length
          verbose: whether to print the progress of extracting feature
        Returns:
          feat: numpy array with shape [N, C]
          ids: numpy array with shape [N]
          cams: numpy array with shape [N]
          im_names: numpy array with shape [N]
          marks: numpy array with shape [N]
        """
        feat, ids, id_ch_seg, cams, tracks,  im_names, marks = [], [], [], [], [], [], []
        done = False
        step = 0
        printed = False
        st = time.time()
        last_time = time.time()
        while not done:
            ims_, im_names_, ids_, cams_, tracks_, samples_masks, marks_, done = self.next_batch()
            if done and ims_ is None:
                break
            feat_ = self.extract_feat_func(ims_, samples_masks)
            feat.append(feat_)
            id_ch_seg.append(ids_)
            ids.append([id_ch.split('_')[0] for id_ch in ids_])
            cams.append(cams_)
            tracks.append(tracks_)
            im_names += list(im_names_)
            step += 1
            marks.append(marks_)

            '''
      print ('ids', ids)
      print ('id_ch', id_ch_seg)
      print ('cams', cams)
      print ('tracks', tracks)
      print ('im names', im_names) 
      print ('marks', marks)
      '''
            if verbose:
                # Print the progress of extracting feature
                total_batches = (self.prefetcher.dataset_size
                                 // self.prefetcher.batch_size + 1)
                if step % 20 == 0:
                    if not printed:
                        printed = True
                    else:
                        # Clean the current line
                        sys.stdout.write("\033[F\033[K")
                    print('{}/{} batches done, +{:.2f}s, total {:.2f}s'
                          .format(step, total_batches,
                                  time.time() - last_time, time.time() - st))
                    last_time = time.time()
        feat = np.vstack(feat)
        ids = np.hstack(ids)
        id_ch_seg = np.hstack(id_ch_seg)
        cams = np.hstack(cams)
        tracks = np.hstack(tracks)
        #im_names = np.hstack(im_names)
        marks = np.hstack(marks)

        feat_dict = {}
        for i, ics in enumerate(id_ch_seg):
            id = ids[i]
            f = feat[i]
            cam = cams[i]
            im_name = im_names[i]
            id_ch = '_'.join(ics.split('_')[0:2])
            mark = marks[i]

            if id_ch not in feat_dict:
                feat_dict[id_ch] = {'id': id, 'feat': [],
                                    'cam': cam, 'mark': mark, 'im_names': []}
            feat_dict[id_ch]['feat'].append(f)
            feat_dict[id_ch]['im_names'] += im_name

        feat, ids, cams, im_names, marks = [], [], [], [], []

        for key in feat_dict:
            f = feat_dict[key]['feat']
            f = np.mean(np.vstack(f), axis=0)
            f = normalize(f, axis=0)

            feat.append(f)
            ids.append(feat_dict[key]['id'])
            cams.append(feat_dict[key]['cam'])
            marks.append(feat_dict[key]['mark'])
            im_names.append(feat_dict[key]['im_names'])
        feat = np.array(feat)
        ids = np.array(ids)
        cams = np.array(cams)
        marks = np.array(marks)
        print(ids, cams, marks, im_names)
        return feat, ids, cams, im_names, marks

    def eval(
            self,
            normalize_feat=True,
            to_re_rank=False,
            pool_type='average',
            verbose=True,
            preload_feature=False):
        """Evaluate using metric CMC and mAP.
        Args:
          normalize_feat: whether to normalize features before computing distance
          to_re_rank: whether to also report re-ranking scores
          pool_type: 'average' or 'max', only for multi-query case
          verbose: whether to print the intermediate information
        """
        #to_re_rank = False
        if preload_feature:
            feat, ids, cams, im_names, marks = pickle.load(
                open('test_preload_feature.pkl'))
        else:
            with measure_time('Extracting feature...', verbose=verbose):
                feat, ids, cams, im_names, marks = self.extract_feat(
                    normalize_feat, verbose)
                #im_names = [x if isinstance(x, str) else x.decode('utf-8') for x in im_names]
            #pickle.dump((feat, ids, cams, im_names, marks), open('test_preload_feature.pkl', 'w'))
        # query, gallery, multi-query indices
        '''
    """
    rearrange query and gallery, use all the images of the same id and cam_id, as query, others as gallery
    """
    print('ids:', ids.shape)
    print('cams:', cams.shape)
    feat_dim = feat.shape[1]
    feat_dict = {}
    for fea, id, cam in zip(feat, ids, cams):
        if id not in feat_dict:
            feat_dict[id] = {}
        if cam not in feat_dict[id]:
            feat_dict[id][cam] = fea
        else:
            feat_dict[id][cam] = np.vstack((feat_dict[id][cam], fea))

    new_ids = []  # the rank of new person ids
    new_feat_matrix = []
    query_ids = [] # choose which cam_id of one person to be query
    query_feats = np.array([])
    gallery_feats = np.array([])
    gallery_ids = []
    query_cams = []
    gallery_cams = []
    for i, p_id in enumerate(sorted(feat_dict)):
        print('p_id:', i)
        new_ids.append(p_id)
        new_feat_matrix.append([])

        print('p_id:', i)
        new_ids.append(p_id)
        new_feat_matrix.append([])
        for j , cam_track_id in enumerate(sorted(feat_dict[p_id])):
            print('cam_track_id:', j)
            #trace_feat = np.mean(feat_dict[p_id][cam_track_id], axis = 0)
            trace_feat = np.copy(feat_dict[p_id][cam_track_id])
            cam_id = cam_track_id[0].zfill(5)
            if j == i % len(feat_dict[p_id]):  #use as query
                if len(query_feats) == 0:
                    query_feats = np.copy(trace_feat)
                else:
                    query_feats = np.vstack((query_feats, trace_feat))
                query_ids.append(p_id)
                #resolve cam_id from cam_track
                query_cams.append(cam_id)
            else: # use as gallery
                if len(gallery_feats) == 0:
                    gallery_feats = np.copy(trace_feat)
                else:
                    gallery_feats = np.vstack((gallery_feats, trace_feat))
                print('gallery:',gallery_feats.shape)
                gallery_ids.append(p_id)
                gallery_cams.append(cam_id)
    if len(query_feats.shape) == 1:
        query_feats = query_feats.reshape(1, query_feats.shape[0])
    if len(gallery_feats.shape) == 1:
        gallery_feats = gallery_feats.reshape(1, gallery_feats.shape[0])
    #dist_mat = compute_dist(query_feats, gallery_feats, type = 'euclidean')
    
      new_ids = []  # the rank of new person ids
      new_feat_matrix = []
      query_ids = [] # choose which cam_id of one person to be query
      query_feats = np.array([])
      gallery_feats = np.array([])
      gallery_ids = []
      query_cams = []
      gallery_cams = []
      for i, p_id in enumerate(sorted(feat_dict)):
          print('p_id:', i)
          new_ids.append(p_id)
          new_feat_matrix.append([])
    
          print('p_id:', i)
          new_ids.append(p_id)
          new_feat_matrix.append([])
          for j , cam_track_id in enumerate(sorted(feat_dict[p_id])):
              print('cam_track_id:', j)
              #trace_feat = np.mean(feat_dict[p_id][cam_track_id], axis = 0)
              trace_feat = np.copy(feat_dict[p_id][cam_track_id])
              cam_id = cam_track_id[0].zfill(5)
              if j == i % len(feat_dict[p_id]):  #use as query
                  if len(query_feats) == 0:
                      query_feats = np.copy(trace_feat)
                  else:
                      query_feats = np.vstack((query_feats, trace_feat))
                  query_ids.append(p_id)
                  #resolve cam_id from cam_track
                  query_cams.append(cam_id)
              else: # use as gallery
                  if len(gallery_feats) == 0:
                      gallery_feats = np.copy(trace_feat)
                  else:
                      gallery_feats = np.vstack((gallery_feats, trace_feat))
                  print('gallery:',gallery_feats.shape)
                  gallery_ids.append(p_id)
                  gallery_cams.append(cam_id)
      if len(query_feats.shape) == 1:
          query_feats = query_feats.reshape(1, query_feats.shape[0])
      if len(gallery_feats.shape) == 1:
          gallery_feats = gallery_feats.reshape(1, gallery_feats.shape[0])
      #dist_mat = compute_dist(query_feats, gallery_feats, type = 'euclidean')
      
      query_ids = np.array(query_ids)
      gallery_ids = np.array(gallery_ids) 
      query_cams = np.array(query_cams)
      gallery_cams = np.array(gallery_cams)
      
      print('query ids', query_ids)
      print('gallery ids', gallery_ids)
      print('query cams', query_cams)
      print('gallery cams', gallery_cams) 
    
    print('query ids', query_ids)
    print('gallery ids', gallery_ids)
    print('query cams', query_cams)
    print('gallery cams', gallery_cams) 

    '''

        q_inds = marks == 0
        g_inds = marks == 1
        mq_inds = marks == 2

        #print (query_ids.shape, gallery_ids.shape, query_cams.shape, marks.shape)
        # A helper function just for avoiding code duplication.
        def compute_score(
                dist_mat,
                query_ids=ids[q_inds],
                gallery_ids=ids[g_inds],
                query_cams=cams[q_inds],
                gallery_cams=cams[g_inds]):
            # Compute mean AP
            print(dist_mat, query_ids, gallery_ids, query_cams, gallery_cams)
            '''
      mAP = mean_ap(
        distmat=dist_mat,
        query_ids=query_ids, gallery_ids=gallery_ids,
        query_cams=query_cams, gallery_cams=gallery_cams)
      '''
            # Compute CMC scores
            '''
      cmc_scores0 = cmc(
        distmat=dist_mat,
        query_ids=query_ids, gallery_ids=gallery_ids,
        query_cams=query_cams, gallery_cams=gallery_cams,
        separate_camera_set=self.separate_camera_set,
        single_gallery_shot=self.single_gallery_shot,
        first_match_break=self.first_match_break,
        topk=10)
      '''
            cmc_scores, mAP = evaluate(
                dist_mat,
                query_ids, gallery_ids,
                query_cams, gallery_cams,
            )
            #raise SystemExit

            '''
      pr_scores = precision_recall(
        distmat=dist_mat,
        query_ids=query_ids, gallery_ids=gallery_ids,
        query_cams=query_cams, gallery_cams=gallery_cams,
        separate_camera_set=self.separate_camera_set,
        thres = 0.8
        )
      '''
            pr_scores = [[], []]
            return mAP, cmc_scores, pr_scores

        def print_scores(mAP, cmc_scores, pr_scores):
            print('[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}]'
                  .format(mAP, *cmc_scores[[0, 4, 9]]))
            for p, r in zip(pr_scores[0], pr_scores[1]):
                print('precision', p, 'recall', r)

        ################
        # Single Query #
        ################

        with measure_time('Computing distance...', verbose=verbose):
            # query-gallery distance
            q_g_dist = compute_dist(
                feat[q_inds], feat[g_inds], type='euclidean')
            #q_g_dist = compute_dist(query_feats, gallery_feats, type = 'euclidean')
        with measure_time('Computing scores...', verbose=verbose):
            mAP, cmc_scores, pr_scores = compute_score(q_g_dist)
            #query_ids = query_ids,
            #gallery_ids = gallery_ids,
            #query_cams = query_cams,
            # gallery_cams = gallery_cams)

        print('{:<30}'.format('Single Query:'), end='')
        print_scores(mAP, cmc_scores, pr_scores)
        s_mAP, s_cmc_scores = mAP, cmc_scores
        return s_mAP, s_cmc_scores, 0, 0, 0, 0, 0, 0
        ###############
        # Multi Query #
        ###############

        mq_mAP, mq_cmc_scores = None, None
        if any(mq_inds):
            mq_ids = ids[mq_inds]
            mq_cams = cams[mq_inds]
            mq_feat = feat[mq_inds]
            unique_mq_ids_cams = defaultdict(list)
            for ind, (id, cam) in enumerate(zip(mq_ids, mq_cams)):
                unique_mq_ids_cams[(id, cam)].append(ind)
            keys = unique_mq_ids_cams.keys()
            assert pool_type in ['average', 'max']
            pool = np.mean if pool_type == 'average' else np.max
            mq_feat = np.stack([pool(mq_feat[unique_mq_ids_cams[k]], axis=0)
                                for k in keys])

            with measure_time('Multi Query, Computing distance...', verbose=verbose):
                # multi_query-gallery distance
                mq_g_dist = compute_dist(
                    mq_feat, feat[g_inds], type='euclidean')

            with measure_time('Multi Query, Computing scores...', verbose=verbose):
                mq_mAP, mq_cmc_scores, pr_scores = compute_score(
                    mq_g_dist,
                    query_ids=np.array(zip(*keys)[0]),
                    gallery_ids=ids[g_inds],
                    query_cams=np.array(zip(*keys)[1]),
                    gallery_cams=cams[g_inds]
                )

            print('{:<30}'.format('Multi Query:'), end='')
            print_scores(mq_mAP, mq_cmc_scores, pr_scores)

        smq_mAP, smq_cmc_scores = mq_mAP, mq_cmc_scores
        rrs_mAP, rrs_cmc_scores = None, None
        rrmq_mAP, rrmq_cmc_scores = None, None
        if to_re_rank:

            ##########################
            # Re-ranked Single Query #
            ##########################

            with measure_time('Re-ranking distance...', verbose=verbose):
                # query-query distance
                q_q_dist = compute_dist(
                    feat[q_inds], feat[q_inds], type='euclidean')
                # gallery-gallery distance
                g_g_dist = compute_dist(
                    feat[g_inds], feat[g_inds], type='euclidean')
                # re-ranked query-gallery distance
                re_r_q_g_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

            with measure_time('Computing scores for re-ranked distance...',
                              verbose=verbose):
                mAP, cmc_scores, pr_scores = compute_score(re_r_q_g_dist)

            print('{:<30}'.format('Re-ranked Single Query:'), end='')
            print_scores(mAP, cmc_scores, pr_scores)

            rrs_mAP, rrs_cmc_scores = mAP, cmc_scores
            smq_mAP, smq_cmc_scores = mq_mAP, mq_cmc_scores
            #########################
            # Re-ranked Multi Query #
            #########################

            if any(mq_inds):
                with measure_time('Multi Query, Re-ranking distance...',
                                  verbose=verbose):
                    # multi_query-multi_query distance
                    mq_mq_dist = compute_dist(
                        mq_feat, mq_feat, type='euclidean')
                    # re-ranked multi_query-gallery distance
                    re_r_mq_g_dist = re_ranking(
                        mq_g_dist, mq_mq_dist, g_g_dist)

                with measure_time(
                    'Multi Query, Computing scores for re-ranked distance...',
                        verbose=verbose):
                    mq_mAP, mq_cmc_scores, pr_scores = compute_score(
                        re_r_mq_g_dist,
                        query_ids=np.array(zip(*keys)[0]),
                        gallery_ids=ids[g_inds],
                        query_cams=np.array(zip(*keys)[1]),
                        gallery_cams=cams[g_inds]
                    )

                print('{:<30}'.format('Re-ranked Multi Query:'), end='')
                print_scores(mq_mAP, mq_cmc_scores, pr_scores)

            rrmq_mAP, rrmq_cmc_scores = mq_mAP, mq_cmc_scores
        # return mAP, cmc_scores, mq_mAP, mq_cmc_scores
        return s_mAP, s_cmc_scores, smq_mAP, smq_cmc_scores, rrs_mAP, rrs_cmc_scores, rrmq_mAP, rrmq_cmc_scores
