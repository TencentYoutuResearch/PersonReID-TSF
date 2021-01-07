~/anaconda2/bin/python -u script/experiment/train_test_lite.py \
-d '([5])' \
--only_test false \
--dataset MARS-pcb \
--trainset_part trainval_avg_mars \
--exp_dir logs/train_video_01/ \
--steps_per_log 21 \
--epochs_per_val 1 \
--batch_size 4 \
--test_batch_size 4 \
--resume false \
--staircase_decay_at_epochs '([15,30,40,50,60,70])' \
--new_params_lr 0.0001 \
--staircase_decay_multiply_factor 0.2 \
--finetuned_params_lr 0.0001 \
--fc_params_lr 0.001 \
--class_balance false \
--random_erasing_prob 0.0 \
--image_save false \
--hsv_jitter_prob 0.0 \
--hsv_jitter_range '(50,15,40)' \
--gaussian_blur_prob 0.1 \
--gaussian_blur_kernel 5 \
--horizontal_crop_ratio 1.0 \
--horizontal_crop_prob  0.0 \
--net stf.model.temp_semantic_att \
--num_stripes 1 \
--local_conv_out_channels 4096 \
--resize_h_w '(384,128)' \
--loss LSR \

# --model_weight_file ../video_reid/logs/run_track_05/6ep_ckpt.pth \