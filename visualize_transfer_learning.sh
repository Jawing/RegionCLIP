# visualize detection results from finetuned detectors on custom images

#################################################################################

# RN50, Finetune (COCO) on custom
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco.yaml \
MODEL.WEIGHTS ./output/model_best_small_110000.pth \
MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5key.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5key.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \


# visualize the prediction json file
python ./tools/visualize_json_results.py \
--input ./output/inference/lvis_instances_results.json \
--output ./output/best_small_test_hw_viz \
--dataset humanware_val_custom_img \
--conf-threshold 0.10 \
--show-unique-boxes \
--max-boxes 100 \
--small-region-px 420 \

#change ood folder images
#rsync -a --delete custom_images_hw_test/ custom_images/
#rsync -a --delete custom_images_basic_test/ custom_images/


# # RN50, Finetune (LVIS) on custom 
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_lvis.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-lvis_rn50.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_P_RN50.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_P_RN50.pth \
# MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \

# visualize the prediction json file
# python ./tools/visualize_json_results.py \
# --input ./output/inference/lvis_instances_results.json \
# --output ./output/custom_ft_lvis_iid_lvis \
# --dataset humanware_val_custom_img \
# --conf-threshold 0.30 \
# --show-unique-boxes \
# --max-boxes 50 \
# --small-region-px 210 \

#################################################################################

# # RN50x4, HUMANWARE (COCO FINAL NN model) on custom
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco.yaml \
# MODEL.WEIGHTS ./models/model_best_large_90000.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50x4_nn5key.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50x4_nn5key.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
# MODEL.CLIP.TEXT_EMB_DIM 640 \
# MODEL.RESNETS.DEPTH 200 \
# MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
# # MODEL.CLIP.MULTIPLY_RPN_SCORE True \
# # MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \
# # need to enable in settings https://github.com/microsoft/RegionCLIP/issues/13

# # visualize the prediction json file
# python ./tools/visualize_json_results.py \
# --input ./output/inference/lvis_instances_results.json \
# --output ./output/best_large_test_basic_viz \
# --dataset humanware_val_custom_img \
# --conf-threshold 0.10 \
# --show-unique-boxes \
# --max-boxes 100 \
# --small-region-px 420 \

#change ood folder images
#rsync -a --delete custom_images_hw_test/ custom_images/
#rsync -a --delete custom_images_basic_test/ custom_images/


# RN50, HUMANWARE (COCO) on custom
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco.yaml \
# MODEL.WEIGHTS ./output/model_final_coco_18000.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_P_RN50.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_P_RN50.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \

# visualize the prediction json file
# python ./tools/visualize_json_results.py \
# --input ./output/inference/lvis_instances_results.json \
# --output ./output/custom_ft_ood_coco \
# --dataset humanware_val_custom_img \
# --conf-threshold 0.70 \
# --show-unique-boxes \
# --max-boxes 50 \
# --small-region-px 210 \

#change ood folder custom_ft_ood_coco
#change ood folder images rsync -a --delete custom_images_ood/ custom_images/


# RN50, HUMANWARE (LVIS) on custom
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_lvis.yaml \
# MODEL.WEIGHTS ./output/model_final_lvis_18000.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_P_RN50.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_P_RN50.pth \
# MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \


# visualize the prediction json file
# python ./tools/visualize_json_results.py \
# --input ./output/inference/lvis_instances_results.json \
# --output ./output/custom_ft_ood_lvis \
# --dataset humanware_val_custom_img \
# --conf-threshold 0.70 \
# --show-unique-boxes \
# --max-boxes 50 \
# --small-region-px 210 \

#################################################################################

# RN50, HUMANWARE (COCO) on test set
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco.yaml \
# MODEL.WEIGHTS ./output/model_final_coco_18000.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_P_RN50.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_P_RN50.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \


# visualize the prediction json file
# python ./tools/visualize_json_results.py \
# --input ./output/inference/coco_instances_results.json \
# --output ./output/custom_ft_coco \
# --dataset humanware_test \
# --conf-threshold 0.70 \
# --show-unique-boxes \
# --max-boxes 50 \
# --small-region-px 210 \

########################################################
# On Humanware dataset (LVIS) RN50x4
# Open-vocabulary detector trained by 866 LVIS base categories, with RegionCLIP (RN50x4) as initialization
# python3 ./tools/train_net.py \
# --eval-only \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-lvis_rn50x4.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_RN50x4.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_RN50x4.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
# MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \
# MODEL.CLIP.TEXT_EMB_DIM 640 \
# MODEL.RESNETS.DEPTH 200 \
# MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
# MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION 18 \
# MODEL.RESNETS.RES2_OUT_CHANNELS 320 \

# visualize the prediction json file
# python ./tools/visualize_json_results.py \
# --input ./output/inference/lvis_instances_results.json \
# --output ./output/custom_ft2 \
# --dataset humanware_val_custom_img \
# --conf-threshold 0.80 \
# --show-unique-boxes \
# --max-boxes 25 \
# --small-region-px 8100 \
#min detected box area 211

########################################################

# Open-vocabulary detector trained by 866 LVIS base categories, with RegionCLIP (RN50x4) as initialization
# python3 ./tools/train_net.py \
# --eval-only \
# --num-gpus 1 \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_custom_img.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-lvis_rn50x4.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb_rn50x4.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb_rn50x4.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
# MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \
# MODEL.CLIP.TEXT_EMB_DIM 640 \
# MODEL.RESNETS.DEPTH 200 \
# MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
# MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION 18 \
# MODEL.RESNETS.RES2_OUT_CHANNELS 320 \

# visualize the prediction json file
# python ./tools/visualize_json_results.py \
# --input ./output/inference/lvis_instances_results.json \
# --output ./output/regions \
# --dataset lvis_v1_val_custom_img \
# --conf-threshold 0.05 \
# --show-unique-boxes \
# --max-boxes 25 \
# --small-region-px 8100\


########################################################

# Open-vocabulary detector trained by 866 LVIS base categories, with RegionCLIP (RN50) as initialization
# python3 ./tools/train_net.py \
# --eval-only \
# --num-gpus 1 \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_custom_img.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-lvis_rn50.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
# MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \

# # visualize the prediction json file
# python ./tools/visualize_json_results.py \
# --input ./output/inference/lvis_instances_results.json \
# --output ./output/regions \
# --dataset lvis_v1_val_custom_img \
# --conf-threshold 0.05 \
# --show-unique-boxes \
# --max-boxes 25 \
# --small-region-px 8100\





