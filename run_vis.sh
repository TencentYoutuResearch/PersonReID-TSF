~/anaconda2/bin/python -u script/experiment/visualize_rank_list_att_track.py \
-d '([0,1,3])' \
--dataset MARS-pcb \
--exp_dir logs/run_vis_att/ \
--batch_size 6 \
--model logs/run_vis_semantic \
--net bpm.model.PCBModel_temp_semantic_att \
--num_stripes 1 \
--local_conv_out_channels 768 \
--resize_h_w '(384,128)' \
--rank_list_size 25 \
--model_weight_file /data1/sevjiang/Projects/video_reid/logs/run_track_05/1ep_ckpt.pth 
#--net bpm.model.PCBModel_temp_att_nonlocal2 \
#--dataset wanda_0811_0819_0824_0826_0827_0901_0904_0905_baili1012_1014 \
#--dataset maxwin_label_p3_0_1 \
#--testset_part test_neural_dist
#--view_pred labels/att_wanda_0811_0819_0824_0826_0827_0901_0904_0905_refine2

#--model pretrained/temp_att_RN_checkpoint.pth.tar-13-0 \
#--dataset wanda_0811_0819_0824_0826_0827_0901_0904_0905_baili1012_1014_track \
