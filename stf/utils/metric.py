"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid) 
reid/evaluation_metrics/ranking.py. Modifications: 
1) Only accepts numpy data input, no torch is involved.
1) Here results of each query can be returned.
2) In the single-gallery-shot evaluation case, the time of repeats is changed 
   from 10 to 100.
"""
from __future__ import absolute_import
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def cmc(
        distmat,
        query_ids=None,
        gallery_ids=None,
        query_cams=None,
        gallery_cams=None,
        topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False,
        average=True):
    """
    Args:
      distmat: numpy array with shape [num_query, num_gallery], the 
        pairwise distance between query and gallery samples
      query_ids: numpy array with shape [num_query]
      gallery_ids: numpy array with shape [num_gallery]
      query_cams: numpy array with shape [num_query]
      gallery_cams: numpy array with shape [num_gallery]
      average: whether to average the results across queries
    Returns:
      If `average` is `False`:
        ret: numpy array with shape [num_query, topk]
        is_valid_query: numpy array with shape [num_query], containing 0's and 
          1's, whether each query is valid or not
      If `average` is `True`:
        numpy array with shape [topk]
    """
    # Ensure numpy array
    assert isinstance(distmat, np.ndarray)
    assert isinstance(query_ids, np.ndarray)
    assert isinstance(gallery_ids, np.ndarray)
    assert isinstance(query_cams, np.ndarray)
    assert isinstance(gallery_cams, np.ndarray)

    m, n = distmat.shape
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros([m, topk])
    is_valid_query = np.zeros(m)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]):
            continue
        is_valid_query[i] = 1
        if single_gallery_shot:
            repeat = 100
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk:
                    break
                if first_match_break:
                    ret[i, k - j] += 1
                    break
                ret[i, k - j] += delta
        num_valid_queries += 1
    # if num_valid_queries == 0:
    #  raise RuntimeError("No valid query")
    ret = ret.cumsum(axis=1)
    if average:
        return np.sum(ret, axis=0) / (num_valid_queries + 1e-7)
    return ret, is_valid_query


def precision_recall(
    distmat,
    query_ids=None,
    gallery_ids=None,
    query_cams=None,
    gallery_cams=None,
    separate_camera_set=False,
    thres=0.6
):
    m, n = distmat.shape
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    num_valid_queries = 0
    recalls = []
    precisions = []
    thres = 0.1
    while thres < 2:
        hit = 0
        n_preds = 0
        n_gt = 0
        for i in range(m):
            # Filter out the same id and same camera
            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                     (gallery_cams[indices[i]] != query_cams[i]))
            if separate_camera_set:
                # Filter out samples from same camera
                valid &= (gallery_cams[indices[i]] != query_cams[i])

            if not np.any(matches[i, valid]):
                continue
            dist = distmat[i, indices[i]]
            #thres = 0.5
            # print matches[i][valid], dist[valid], min(dist)
            match = matches[i][valid][0]
            score = dist[valid][0]
            # print match, score
            if match and score < thres:
                hit += 1
            if score < thres:
                n_preds += 1
            n_gt += 1
#    print hit, n_preds, n_gt
        precision = hit * 1.0 / (n_preds + 1e-5)
        recall = hit * 1.0 / n_gt
        recalls.append(recall)
        precisions.append(precision)
        thres += 0.01
    ret_recall = recalls
    ret_precision = precisions

    return [ret_precision, ret_recall]


def mean_ap(
        distmat,
        query_ids=None,
        gallery_ids=None,
        query_cams=None,
        gallery_cams=None,
        average=True):
    """
    Args:
      distmat: numpy array with shape [num_query, num_gallery], the 
        pairwise distance between query and gallery samples
      query_ids: numpy array with shape [num_query]
      gallery_ids: numpy array with shape [num_gallery]
      query_cams: numpy array with shape [num_query]
      gallery_cams: numpy array with shape [num_gallery]
      average: whether to average the results across queries
    Returns:
      If `average` is `False`:
        ret: numpy array with shape [num_query]
        is_valid_query: numpy array with shape [num_query], containing 0's and 
          1's, whether each query is valid or not
      If `average` is `True`:
        a scalar
    """
    # -------------------------------------------------------------------------
    # The behavior of method `sklearn.average_precision` has changed since version
    # 0.19.
    # Version 0.18.1 has same results as Matlab evaluation code by Zhun Zhong
    # (https://github.com/zhunzhong07/person-re-ranking/
    # blob/master/evaluation/utils/evaluation.m) and by Liang Zheng
    # (http://www.liangzheng.org/Project/project_reid.html).
    # My current awkward solution is sticking to this older version.
    import sklearn
    cur_version = sklearn.__version__
    required_version = '0.18.1'
    if cur_version != required_version:
        print('User Warning: Version {} is required for package scikit-learn, '
              'your current version is {}. '
              'As a result, the mAP score may not be totally correct. '
              'You can try `pip uninstall scikit-learn` '
              'and then `pip install scikit-learn=={}`'.format(
                  required_version, cur_version, required_version))
    # -------------------------------------------------------------------------

    # Ensure numpy array
    assert isinstance(distmat, np.ndarray)
    assert isinstance(query_ids, np.ndarray)
    assert isinstance(gallery_ids, np.ndarray)
    assert isinstance(query_cams, np.ndarray)
    assert isinstance(gallery_cams, np.ndarray)

    m, n = distmat.shape

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = np.zeros(m)
    is_valid_query = np.zeros(m)
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true):
            continue
        is_valid_query[i] = 1
        aps[i] = average_precision_score(y_true, y_score)
        #print (aps[i])
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    if average:
        return float(np.sum(aps)) / np.sum(is_valid_query)
    return aps, is_valid_query
