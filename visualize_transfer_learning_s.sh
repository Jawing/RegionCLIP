# visualize detection results from finetuned detectors on custom images

#################################################################################

# RN50, (COCO) on custom
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco.yaml \
MODEL.WEIGHTS ./output/model_best_small_110000.pth \
MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5key.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5key.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \

# MODEL.CLIP.MULTIPLY_RPN_SCORE True \
# MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \
# need to enable in settings https://github.com/microsoft/RegionCLIP/issues/13

# visualize the prediction json file
python ./tools/visualize_json_results.py \
--input ./output/inference/lvis_instances_results.json \
--output ./output/custom_coco_s \
--dataset humanware_val_custom_img \
--conf-threshold 0.10 \
--show-unique-boxes \
--max-boxes 100 \
--small-region-px 420 \

#change ood folder images
#rsync -a --delete custom_images_hw_test/ custom_images/
#rsync -a --delete custom_images_basic_test/ custom_images/


# # RN50, HUMANWARE (LVIS) on custom 
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_lvis.yaml \
# MODEL.WEIGHTS ./output/model_final_lvis_18000.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5key.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5key.pth \
# MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \

# visualize the prediction json file
# python ./tools/visualize_json_results.py \
# --input ./output/inference/lvis_instances_results.json \
# --output ./output/custom_lvis_s \
# --dataset humanware_val_custom_img \
# --conf-threshold 0.10 \
# --show-unique-boxes \
# --max-boxes 100 \
# --small-region-px 420 \
#min detected box area 211