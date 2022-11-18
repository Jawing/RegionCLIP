# Extract region features for a folder of images

# RN50, COCO, GT
python3 ./tools/extract_region_features_full.py \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco_rf_gt.yaml \
MODEL.WEIGHTS ./output/model_best_s_110000_nkc.pth \
MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
OUTPUT_DIR ./tools/vis_regions \

#define input dataset in DATASETS.TEST: ("humanware_test_custom",)

# # RN50, COCO, RPN
# python3 ./tools/extract_region_features_full.py \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco_rf.yaml \
# MODEL.WEIGHTS ./output/model_best_s_110000_nkc.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
# OUTPUT_DIR ./tools/vis_regions \
