from __future__ import print_function

import sys

sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel

import time
import os.path as osp
from tensorboardX import SummaryWriter
import numpy as np
import argparse
import os
import importlib

import random

from stf.dataset import create_dataset

from stf.utils.utils import time_str
from stf.utils.utils import str2bool
from stf.utils.utils import may_set_mode
from stf.utils.utils import load_state_dict
from stf.utils.utils import load_ckpt
from stf.utils.utils import save_ckpt
from stf.utils.utils import set_devices
from stf.utils.utils import AverageMeter
from stf.utils.utils import to_scalar
from stf.utils.utils import ReDirectSTD
from stf.utils.utils import set_seed
from stf.utils.utils import adjust_lr_staircase
from stf.utils.distance import normalize
from stf.utils.loss_func import LSR

import torch.nn.functional as F
import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
 

class Config(object):
    def __init__(self):
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
        parser.add_argument('-r', '--run', type=int, default=1)
        parser.add_argument('--set_seed', type=str2bool, default=False)
        parser.add_argument('--dataset', type=str, default='market1501')
        parser.add_argument('--trainset_part', type=str, default='trainval',
                            choices=['trainval', 'train', 'trainval_avg_mars'])
        parser.add_argument('--testset_part', type=str, default='test',
                            choices=['test', 'test_avg'])
        
        parser.add_argument('--resize_h_w', type=eval, default=(384, 128))
        # These several only for training set
        parser.add_argument('--crop_prob', type=float, default=0)
        parser.add_argument('--crop_ratio', type=float, default=1)
        parser.add_argument('--mirror', type=str2bool, default=True)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--test_batch_size', type=int, default=16)
        parser.add_argument('--random_erasing_prob', type=float, default=0)
        parser.add_argument('--fix_hw_rate', type=str2bool, default=False)
        parser.add_argument('--image_save', type=str2bool, default=False)
        
        parser.add_argument('--hsv_jitter_prob', type=float, default=0)
        parser.add_argument('--hsv_jitter_range', type=eval, default=(50, 20, 40))
        
        parser.add_argument('--gaussian_blur_prob', type=float, default=0)
        parser.add_argument('--gaussian_blur_kernel', type=int, default=7)
        
        parser.add_argument('--horizontal_crop_prob', type=float, default=0)
        parser.add_argument('--horizontal_crop_ratio', type=float, default=0.4)
        
        parser.add_argument('--log_to_file', type=str2bool, default=True)
        parser.add_argument('--steps_per_log', type=int, default=20)
        parser.add_argument('--epochs_per_val', type=int, default=1)
        
        parser.add_argument('--last_conv_stride', type=int, default=1, choices=[1, 2])
        # When the stride is changed to 1, we can compensate for the receptive field
        # using dilated convolution. However, experiments show dilated convolution is useless.
        parser.add_argument('--last_conv_dilation', type=int, default=1, choices=[1, 2])
        parser.add_argument('--num_stripes', type=int, default=6)
        parser.add_argument('--local_conv_out_channels', type=int, default=256)
        
        parser.add_argument('--only_test', type=str2bool, default=False)
        parser.add_argument('--resume', type=str2bool, default=False)
        parser.add_argument('--exp_dir', type=str, default='')
        parser.add_argument('--model_weight_file', type=str, default='')
        
        parser.add_argument('--new_params_lr', type=float, default=0.1)
        parser.add_argument('--finetuned_params_lr', type=float, default=0.01)
        parser.add_argument('--fc_params_lr', type=float, default=0.1)
        parser.add_argument('--double_params_lr', type=float, default=0.1)
        parser.add_argument('--staircase_decay_at_epochs',
                        type=eval, default=(25,))
        parser.add_argument('--staircase_decay_multiply_factor',
                            type=float, default=0.1)
        parser.add_argument('--total_epochs', type=int, default=200)
        parser.add_argument('--class_balance', type=str2bool, default=False)
        parser.add_argument('--camera_weight', type=str2bool, default=False)
        parser.add_argument('--net', type=str, default='bpm.model.temp_semantic_att')
        parser.add_argument('--sep_norm', type=str2bool, default=False)
        parser.add_argument('--loss', type=str, default='softmax', 
                choices=['softmax', 'LSR'])
        parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'])
        parser.add_argument('--fix_all_but_fc', type=str2bool, default=False)
        parser.add_argument('--view_pred', type=str, default='')
         parser.add_argument('--view_filter', type=str2bool, default=False)
        args = parser.parse_args()
        
        # gpu ids
        self.sys_device_ids = args.sys_device_ids
        
        # If you want to make your results exactly reproducible, you have
        # to fix a random seed.
        if args.set_seed:
            self.seed = 1
        else:
            self.seed = None
        
        # The experiments can be run for several times and performances be averaged.
        # `run` starts from `1`, not `0`.
        self.run = args.run
        
        ###########
        # Dataset #
        ###########
        
        # If you want to make your results exactly reproducible, you have
        # to also set num of threads to 1 during training.
        if self.seed is not None:
            self.prefetch_threads = 1
        else:
            self.prefetch_threads = 1
        
        self.dataset = args.dataset
        self.trainset_part = args.trainset_part
        self.testset_part = args.testset_part
        self.class_balance = args.class_balance
        self.camera_weight = args.camera_weight 
        self.net = args.net
        self.sep_norm = args.sep_norm
        self.loss = args.loss
        # Image Processing
        
        # Just for training set
        self.crop_prob = args.crop_prob
        self.crop_ratio = args.crop_ratio
        self.resize_h_w = args.resize_h_w
        self.random_erasing_prob = args.random_erasing_prob
        self.hsv_jitter_prob = args.hsv_jitter_prob
        self.hsv_jitter_range = args.hsv_jitter_range
        self.gaussian_blur_prob = args.gaussian_blur_prob
        self.gaussian_blur_kernel = args.gaussian_blur_kernel
        self.horizontal_crop_prob = args.horizontal_crop_prob
        self.horizontal_crop_ratio = args.horizontal_crop_ratio
        self.fix_all_but_fc = args.fix_all_but_fc
        self.view_pred = ''
        # Whether to scale by 1/255
        self.scale_im = True
        self.im_mean = [0.486, 0.459, 0.408]
        self.im_std = [0.229, 0.224, 0.225]
        
        self.train_mirror_type = 'random' if args.mirror else None
        self.train_batch_size = args.batch_size
        self.train_final_batch = False
        self.train_shuffle = True
        
        self.test_mirror_type = None
        self.test_batch_size = args.test_batch_size
        self.test_final_batch = True
        self.test_shuffle = False
        self.fix_hw_rate = args.fix_hw_rate
        self.image_save = args.image_save
        self.optim = args.optim
        
        
        if self.testset_part == 'test_avg' or self.trainset_part == 'trainval_avg_mars' or self.trainset_part == 'trainval_avg_mars_balance':
            self.ExtractFeature = ExtractFeatureAvg
        else:
            self.ExtractFeature = ExtractFeature
        
        
        dataset_kwargs = dict(
          name=self.dataset,
          resize_h_w=self.resize_h_w,
          scale=self.scale_im,
          im_mean=self.im_mean,
          im_std=self.im_std,
          batch_dims='NCHW',
          att_list = self.view_pred,
          num_prefetch_threads=self.prefetch_threads)
        
        prng = np.random
        if self.seed is not None:
            prng = np.random.RandomState(self.seed)
        self.train_set_kwargs = dict(
          part=self.trainset_part,
          class_balance = self.class_balance,
          camera_weight = self.camera_weight,  
          batch_size=self.train_batch_size,      
          final_batch=self.train_final_batch,
          shuffle=self.train_shuffle,
          crop_prob=self.crop_prob,
          crop_ratio=self.crop_ratio,
          mirror_type=self.train_mirror_type,
          prng=prng,
          random_erasing_prob=self.random_erasing_prob,
          fix_hw_rate = self.fix_hw_rate,
          image_save = self.image_save, 
          hsv_jitter_prob = self.hsv_jitter_prob, 
          hsv_jitter_range = self.hsv_jitter_range, 
          gaussian_blur_prob = self.gaussian_blur_prob, 
          gaussian_blur_kernel = self.gaussian_blur_kernel, 
          horizontal_crop_prob=self.horizontal_crop_prob,
          horizontal_crop_ratio=self.horizontal_crop_ratio, 
        
        
        self.train_set_kwargs.update(dataset_kwargs)
        
        prng = np.random
        if self.seed is not None:
            prng = np.random.RandomState(self.seed)
        self.test_set_kwargs = dict(
          part=self.testset_part,
          batch_size=self.test_batch_size,
          final_batch=self.test_final_batch,
          shuffle=self.test_shuffle,
          mirror_type=self.test_mirror_type,
          fix_hw_rate = self.fix_hw_rate,
          prng=prng)
        
        if self.trainset_part == 'trainval_avg_mars':
            self.test_set_kwargs['part'] = 'test_avg_mars' 
        
        self.test_set_kwargs.update(dataset_kwargs)
        
        ###############
        # ReID Model  #
        ###############
        
        # The last block of ResNet has stride 2. We can set the stride to 1 so that
        # the spatial resolution before global pooling is doubled.
        self.last_conv_stride = args.last_conv_stride
        # When the stride is changed to 1, we can compensate for the receptive field
        # using dilated convolution. However, experiments show dilated convolution is useless.
        self.last_conv_dilation = args.last_conv_dilation
        # Number of stripes (parts)
        self.num_stripes = args.num_stripes
        # Output channel of 1x1 conv
        self.local_conv_out_channels = args.local_conv_out_channels
        
        #############
        # Training  #
        #############
        
        self.momentum = 0.9
        self.weight_decay = 0.0005
        
        # Initial learning rate
        self.new_params_lr = args.new_params_lr
        self.finetuned_params_lr = args.finetuned_params_lr
        self.fc_params_lr = args.fc_params_lr
        self.double_params_lr = args.double_params_lr
        self.staircase_decay_at_epochs = args.staircase_decay_at_epochs
        self.staircase_decay_multiply_factor = args.staircase_decay_multiply_factor
        # Number of epochs to train
        self.total_epochs = args.total_epochs
        
        # How often (in epochs) to test on val set.
        self.epochs_per_val = args.epochs_per_val
        
        # How often (in batches) to log. If only need to log the average
        # information for each epoch, set this to a large value, e.g. 1e10.
        self.steps_per_log = args.steps_per_log
        
        # Only test and without training.
        self.only_test = args.only_test
        
        self.resume = args.resume
        
        #######
        # Log #
        #######
        
        # If True,
        # 1) stdout and stderr will be redirected to file,
        # 2) training loss etc will be written to tensorboard,
        # 3) checkpoint will be saved
        self.log_to_file = args.log_to_file
        
        # The root dir of logs.
        if args.exp_dir == '':
            self.exp_dir = osp.join(
              'exp/train',
              '{}'.format(self.dataset),
              'run{}'.format(self.run),
            )
        else:
            self.exp_dir = args.exp_dir
        
        self.stdout_file = osp.join(
          self.exp_dir, 'stdout_{}.txt'.format(time_str()))
        self.stderr_file = osp.join(
          self.exp_dir, 'stderr_{}.txt'.format(time_str()))
        
        # Saving model weights and optimizer states, for resuming.
        self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')
        # Just for loading a pretrained model; no optimizer states is needed.
        self.model_weight_file = args.model_weight_file
    
class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT, sep_norm = False):
      self.model = model
      self.TVT = TVT
      self.sep_norm = sep_norm
      
  def __call__(self, ims, is_front = None):
      old_train_eval_model = self.model.training
      # Set eval mode.
      # Force all BN layers to use global mean and variance, also disable
      # dropout.
      self.model.eval()
      ims = Variable(self.TVT(torch.from_numpy(ims).float()))
      if hasattr(self.model.module, 'fc_list'):
        local_feat_list, logits_list = self.model(ims)
      else:
        local_feat_list = self.model(ims) 
      feat = [lf.data.cpu().numpy() for lf in local_feat_list]
      feat = np.concatenate(feat, axis=1) # Restore the model to its old train/eval mode.
      self.model.train(old_train_eval_model)
      
      return feat

class ExtractFeatureAvg(object):
    """A function to be called in the val/test set, to extract features.
    Args:
        TVT: A callable to transfer images to specific device.
    """

    def __init__(self, model, TVT, sep_norm = False):
        self.model = model
        self.TVT = TVT
        self.sep_norm = sep_norm
        self.elapsed = 0
        self.n = 0

    def __call__(self, ims, sample_masks):
        old_train_eval_model = self.model.training
        # Set eval mode.
        # Force all BN layers to use global mean and variance, also disable
        # dropout.
        self.model.eval()
        #ims has size of [N_track * Track_size * C * H * W]
        ims = Variable(self.TVT(torch.from_numpy(ims).float()))
        mask = Variable(self.TVT(torch.from_numpy(sample_masks).float()))

        start = time.clock()

        try:
          local_feat_list, logits_list = self.model(ims, sample_mask=mask)
        except:
          local_feat_list = self.model(ims, sample_mask=mask)

        elapsed = (time.clock() - start)
        self.elapsed  = (self.elapsed * self.n + elapsed) / (self.n + 1)
        print ("time", self.elapsed, 'n frames', ims.size(1), self.n + 1)
        self.n += 1

        if self.sep_norm:
          feat = [normalize(lf.data.cpu().numpy(), axis = 1) for lf in local_feat_list]
        else:
          feat = [lf.data.cpu().numpy() for lf in local_feat_list]
        feat = np.concatenate(feat, axis=1)
        # Restore the model to its old train/eval mode.
        self.model.train(old_train_eval_model)
        return feat
  
def test(load_model_weight=False):
    if load_model_weight:
        if cfg.model_weight_file != '':
            map_location = (lambda storage, loc: storage)
            sd = torch.load(cfg.model_weight_file, map_location=map_location)
            #load_state_dict(model, sd)
            loaded = sd['state_dicts'][0]
            load_state_dict(model, loaded)
            print('Loaded model weights from {}'.format(cfg.model_weight_file))
        else:
            load_ckpt(modules_optims, cfg.ckpt_file)
    
    for test_set, name in zip(test_sets, test_set_names):
        test_set.set_feat_func(cfg.ExtractFeature(model_w, TVT, cfg.sep_norm))
        print('\n=========> Test on dataset: {} <=========\n'.format(name))
        s_mAP, s_cmc_scores, smq_mAP, smq_cmc_scores, rrs_mAP, rrs_cmc_scores, rrmq_mAP, rrmq_cmc_scores = test_set.eval(
          normalize_feat=True,
          to_re_rank=False,
          verbose=True)
    print ("test finished!!")
    return s_mAP, s_cmc_scores, smq_mAP, smq_cmc_scores, rrs_mAP, rrs_cmc_scores, rrmq_mAP, rrmq_cmc_scores

def test_new(load_model_weight=False):
    if load_model_weight:
        if (cfg.model_weight_file != '') and os.path.isfile(cfg.model_weight_file):
            load_ckpt(modules_optims, cfg.model_weight_file)
            '''
            map_location = (lambda storage, loc: storage)
            sd = torch.load(cfg.model_weight_file, map_location=map_location)
            #load_state_dict(model, sd)
            try:
                loaded = sd['state_dicts'][0]
            except:
                loaded = sd['state_dict']
            load_state_dict(model, loaded)
            print('Loaded model weights from {}'.format(cfg.model_weight_file))
            '''
            for test_set, name in zip(test_sets, test_set_names):
                test_set.set_feat_func(cfg.ExtractFeature(model_w, TVT, cfg.sep_norm))
                print('\n=========> Test on dataset: {} <=========\n'.format(name))
                s_mAP, s_cmc_scores, smq_mAP, smq_cmc_scores, rrs_mAP, rrs_cmc_scores, rrmq_mAP, rrmq_cmc_scores = test_set.eval(normalize_feat=True, to_re_rank = False, verbose=True)
                print ("finish!!")
                return s_mAP, s_cmc_scores, smq_mAP, smq_cmc_scores, rrs_mAP, rrs_cmc_scores, rrmq_mAP, rrmq_cmc_scores
        elif (cfg.model_weight_file != '') and (not os.path.isfile(cfg.model_weight_file)):
            # load_ckpt(modules_optims, cfg.ckpt_file)
            map_cmc_txt = open(osp.join(cfg.exp_dir, 'map_cmc.txt'), 'w')
            file_list = os.listdir(cfg.model_weight_file)
            for file1 in sorted(file_list):
                if file1.endswith('.pth') or 'pth' in file1:
                    map_location = (lambda storage, loc: storage)
                    model_weight_file2 = cfg.model_weight_file + '/' + file1
                    sd = torch.load(model_weight_file2, map_location=map_location)
                    #load_state_dict(model, sd)
                    #loaded = sd['state_dicts'][0]
                    try:
                        loaded = sd['state_dicts'][0]
                    except:
                        loaded = sd['state_dict']
                    load_state_dict(model, loaded)
                    print('Loaded model weights from {}'.format(model_weight_file2))
                    for test_set, name in zip(test_sets, test_set_names):
                        test_set.set_feat_func(cfg.ExtractFeature(model_w, TVT, cfg.sep_norm))
                        print('\n=========> Test on dataset: {} <=========\n'.format(name))
                        s_mAP, s_cmc_scores, smq_mAP, smq_cmc_scores, rrs_mAP, rrs_cmc_scores, rrmq_mAP, rrmq_cmc_scores = test_set.eval(normalize_feat=True,to_re_rank=False,verbose=True)
                        #map_cmc_txt.write("Epoch " + file1 + "\n")
                        #map_cmc_txt.write(str(s_mAP) + " " + str(s_cmc_scores[0]) + " " + str(rrs_mAP) + " " + str(rrs_cmc_scores[0]) +" "+str(rrmq_mAP) + str(rrmq_cmc_scores[0]) +"\n")
            return s_mAP, s_cmc_scores, smq_mAP, smq_cmc_scores, rrs_mAP, rrs_cmc_scores, rrmq_mAP, rrmq_cmc_scores
		

def main():
    import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    #torch.cuda.set_device(0)
    cfg = Config()
    if not os.path.exists(cfg.exp_dir):
        os.mkdir(cfg.exp_dir)
    # Redirect logs to both console and file.
    if cfg.log_to_file:
        ReDirectSTD(cfg.stdout_file, 'stdout', False)
        ReDirectSTD(cfg.stderr_file, 'stderr', False)
        
    # Lazily create SummaryWriter
    writer = None
    
    TVT, TMO = set_devices(cfg.sys_device_ids)
    
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    # Dump the configurations to log.
    import pprint
    print('-' * 60)
    print('cfg.__dict__')
    pprint.pprint(cfg.__dict__)
    print('-' * 60)
    
    ###########
    # Dataset #
    ###########
    
    train_set = create_dataset(**cfg.train_set_kwargs)
    #train_set_adversery = create_dataset(**cfg.train_set_kwargs)
    #train_set_adversery.class_balance = True
    
    num_classes = len(train_set.ids2labels)
    # The combined dataset does not provide val set currently.
    
    test_sets = []
    test_set_names = []
    if cfg.dataset == 'combined':
        for name in ['market1501', 'cuhk03', 'duke']:
            cfg.test_set_kwargs['name'] = name
            test_sets.append(create_dataset(**cfg.test_set_kwargs))
            test_set_names.append(name)
    else:
        test_sets.append(create_dataset(**cfg.test_set_kwargs))
        test_set_names.append(cfg.dataset)
    
    ###########
    # Models  #
    ###########
    print ('inital net:', cfg.net)
    Model = importlib.import_module(cfg.net, package=None).Model
    model = Model(
      last_conv_stride=cfg.last_conv_stride,
      num_stripes=cfg.num_stripes,
      local_conv_out_channels=cfg.local_conv_out_channels,
      num_classes=num_classes
    )
    
    # Model wrapper
    model_w = DataParallel(model)
    #############################
    # Criteria and Optimizers   #
    #############################
    
    criterion = torch.nn.CrossEntropyLoss()
    criterion_lsr = LSR()
    
    # To finetune from ImageNet weights
    try:
        finetuned_params = list(model.base.parameters())
    except:
        finetuned_params = list(model.base0.parameters()) + list(model.base1.parameters())
    # To train from scratch
    new_params = [p for n, p in model.named_parameters()
                  if not n.startswith('base') and not n.startswith('fc_list') and not n.startswith('fc_dr') and 'double' not in n]
    fc_params =  [p for n, p in model.named_parameters()
                  if n.startswith('fc_list')]
    dr_params =  [p for n, p in model.named_parameters()
                  if n.startswith('fc_dr')]
    double_params =  [p for n, p in model.named_parameters()
                  if 'double' in n]
    
    param_groups = [{'params': finetuned_params, 'lr': cfg.finetuned_params_lr},
                    {'params': new_params, 'lr': cfg.new_params_lr},
                    {'params': fc_params, 'lr': cfg.fc_params_lr}, 
                    {'params': dr_params, 'lr': cfg.new_params_lr * 10}, 
                    {'params': double_params, 'lr': cfg.double_params_lr}, 
                    ]
    
    if cfg.optim == 'sgd':
        optimizer = optim.SGD(
          param_groups,
          momentum=cfg.momentum,
          weight_decay=cfg.weight_decay, 
          nesterov = True)
    elif cfg.optim =='adam':
        optimizer = optim.Adam(
          param_groups,
          )
    # Bind them together just to save some codes in the following usage.
    modules_optims = [model, optimizer]
    
    ################################
    # May Resume Models and Optims #
    ################################
    if cfg.resume:
        resume_ep, scores = load_ckpt(modules_optims, cfg.ckpt_file)
    
    # May Transfer Models and Optims to Specified Device. Transferring optimizer
    # is to cope with the case when you load the checkpoint to a new device.
    TMO(modules_optims)
    
    ########
    # Test #
    ########
    if cfg.only_test:
        print ("only test")
        test_new(load_model_weight=True)
        return
    
    ############
    # Training #
    ############
    
    #start_ep = resume_ep if cfg.resume else 0
    start_ep = 0
    map_cmc_txt = open(osp.join(cfg.exp_dir, 'map_cmc.txt'), 'w')
    for ep in range(start_ep, cfg.total_epochs):
        if cfg.fix_all_but_fc and ep < 20:
            adjust_lr_staircase(
            optimizer.param_groups,
            [0, 0, 0.1],
            ep + 1,
            cfg.staircase_decay_at_epochs,
            cfg.staircase_decay_multiply_factor)
        
        # Adjust Learning Rate
        else:
            adjust_lr_staircase(
            optimizer.param_groups,
            [cfg.finetuned_params_lr, cfg.new_params_lr, cfg.fc_params_lr, cfg.new_params_lr * 10, cfg.double_params_lr],
            ep + 1,
            cfg.staircase_decay_at_epochs,
            cfg.staircase_decay_multiply_factor)
        
        may_set_mode(modules_optims, 'train')
        
        # For recording loss
        loss_meter = AverageMeter()
        
        ep_st = time.time()
        step = 0
        epoch_done = False
        while not epoch_done:
            step += 1
            step_st = time.time()
            
            is_front = [None]
            if cfg.trainset_part == 'trainval_avg_mars':
                ims, im_names, labels, mirrored, epoch_done, sample_mask = train_set.next_batch()
                sample_mask = Variable(TVT(torch.from_numpy(sample_mask).float()))
            else:
                ims, im_names, labels, mirrored, is_front, epoch_done = train_set.next_batch()
            
            ims_var = Variable(TVT(torch.from_numpy(ims).float()))
            labels_var = Variable(TVT(torch.from_numpy(labels).long()))
            
            if cfg.trainset_part == 'trainval_avg_mars':
                feat_s_list, logits_list = model_w(ims_var, sample_mask)
            else:
                feat_s_list, logits_list = model_w(ims_var)
            
            if cfg.loss == 'softmax':
                loss = torch.sum(
                  torch.stack([criterion(logits, labels_var) for logits in logits_list]))
            elif cfg.loss == 'LSR':    
                loss = torch.sum(
                    torch.stack([criterion_lsr(logits, labels_var) for logits in logits_list]))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ############
            # Step Log #
            ############
            loss_meter.update(to_scalar(loss))
            
            if step % cfg.steps_per_log == 0:
                log = '\tStep {}/Ep {}, {:.2f}s, loss {:.4f}'.format(
                  step, ep + 1, time.time() - step_st, loss_meter.val)
                print(log)
                
            if step % 5000 == 0:
                s_mAP, s_cmc_scores, smq_mAP, smq_cmc_scores, rrs_mAP, rrs_cmc_scores, rrmq_mAP, rrmq_cmc_scores = test(load_model_weight=False)
                if cfg.log_to_file:
                  save_ckpt_path = osp.join(cfg.exp_dir, str(ep+ 1) + 'ep_' + 'step' + str(step) +'_ckpt.pth')
                  save_ckpt(modules_optims, ep + 1, 0, save_ckpt_path) #cfg.ckpt_file
        
        #############
        # Epoch Log #
        #############
        if ep % 5 == 0:
            log = 'Ep {}, {:.2f}s, loss {:.4f}'.format(
              ep + 1, time.time() - ep_st, loss_meter.avg)
            print(log)
            
            ##########################
            # Test on Validation Set #
            ##########################
            
            # save ckpt  modified by jf
            if cfg.log_to_file:
                save_ckpt_path = osp.join(cfg.exp_dir, str(ep+ 1) + 'ep_ckpt.pth')
                save_ckpt(modules_optims, ep + 1, 0, save_ckpt_path) #cfg.ckpt_file
                
            ########
            # Test #
            ########
            s_mAP, s_cmc_scores, smq_mAP, smq_cmc_scores, rrs_mAP, rrs_cmc_scores, rrmq_mAP, rrmq_cmc_scores = test(load_model_weight=False)
            print ("Input finish!!!")
    
if __name__ == '__main__':
    main()
