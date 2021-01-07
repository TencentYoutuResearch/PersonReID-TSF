from .TestSet_avg_mars import TestSetAvgMARS
from .TrainSet_avg_mars import TrainSetAvgMARS
from ..utils.dataset_utils import parse_im_name
from ..utils.utils import load_pickle
import os.path as osp

import numpy as np

ospj = osp.join
ospeu = osp.expanduser


def create_dataset(
        name='market1501',
        part='trainval',
        **kwargs):
    # assert name in ['yonghui_label_check', 'yonghui_before_label','market1501', 'cuhk03', 'duke', 'combined', 'yonghui_label', 'baili_before_label'], \
    #  "Unsupported Dataset {}".format(name)
    print(part, 'foooooo')
    assert part in ['trainval_avg_mars', 'test_avg_mars'], \
        "Unsupported Dataset Part {}".format(part)

    ########################################
    # Specify Directory and Partition File #
    ########################################
    im_dir = ospeu(
        '/data1/sevjiang/datasets/pcb-format/{}/images'.format(name))
    partition_file = ospeu(
        '/data1/sevjiang/datasets/pcb-format/{}/partitions.pkl'.format(name))
    ##################
    # Create Dataset #
    ##################

    # Use standard Market1501 CMC settings for all datasets here.
    cmc_kwargs = dict(separate_camera_set=False,
                      single_gallery_shot=False,
                      first_match_break=True)

    partitions = load_pickle(partition_file)

    if part == 'trainval_avg_mars':
        im_names = partitions['trainval_im_names'.format(part)]
        ids2labels = partitions['trainval_ids2labels']

        ret_set = TrainSetAvgMARS(
            im_dir=im_dir,
            im_names=im_names,
            ids2labels=ids2labels,
            **kwargs)

    elif part == 'test_avg_mars':
        im_names = partitions['test_im_names']
        marks = partitions['test_marks']
        kwargs.update(cmc_kwargs)

        ret_set = TestSetAvgMARS(
            im_dir=im_dir,
            im_names=im_names,
            marks=marks,
            **kwargs)

    if part in ['trainval', 'train', 'trainval_avg', 'trainval_view', 'trainval_avg2', 'trainval_avg_bbox', 'trainval_bbox', 'trainval_avg_mars', 'trainval_avg_new', 'trainval_avg_mars_balance', 'trainval_global_ohem', 'trainval_triplet']:
        num_ids = len(ids2labels)
    elif part in ['val', 'test', 'test_avg', 'test_avg_att_vis', 'test_view', 'test_neural_dist', 'test_avg2', 'test_bbox', 'test_avg_mars', 'test_avg_new']:
        ids = [parse_im_name(n, 'id') for n in im_names]
        num_ids = len(list(set(ids)))
        num_query = np.sum(np.array(marks) == 0)
        num_gallery = np.sum(np.array(marks) == 1)
        num_multi_query = np.sum(np.array(marks) == 2)
    # Print dataset information
    print('-' * 40)
    print('{} {} set'.format(name, part))
    print('-' * 40)
    print('NO. Images: {}'.format(len(im_names)))
    print('NO. IDs: {}'.format(num_ids))
    try:
        print('NO. Query Images: {}'.format(num_query))
        print('NO. Gallery Images: {}'.format(num_gallery))
        print('NO. Multi-query Images: {}'.format(num_multi_query))
    except:
        pass

    print('-' * 40)

    return ret_set
