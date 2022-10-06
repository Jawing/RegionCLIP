# visualize detection results from finetuned detectors on custom images

#################################################################################
# RN50, Finetune (COCO) on custom
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco_c.yaml \
MODEL.WEIGHTS ./output/model_best_small_110000.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \

#note to do zsinf or openset test set MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH to a different embedding

# MODEL.CLIP.MULTIPLY_RPN_SCORE True \
# MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \
# need to enable in settings https://github.com/microsoft/RegionCLIP/issues/13

# visualize coco json file for dataset test images
python ./tools/visualize_json_results.py \
--input ./output/inference/coco_instances_results.json \
--output ./output/custom_coco_c \
--dataset humanware_val_custom_img \
--conf-threshold 0.10 \
--show-unique-boxes \
--max-boxes 100 \
--small-region-px 420 \

# # visualize lvis json file for custom images
# python ./tools/visualize_json_results.py \
# --input ./output/inference/lvis_instances_results.json \
# --output ./output/custom_coco_c \
# --dataset humanware_val_custom_img \
# --conf-threshold 0.10 \
# --show-unique-boxes \
# --max-boxes 100 \
# --small-region-px 420 \
