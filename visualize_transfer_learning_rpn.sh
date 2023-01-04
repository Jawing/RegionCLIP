# visualize detection results from finetuned detectors on custom images

# ################################################################################
# # RN50, Finetune (COCO) on custom
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco_rpn.yaml \
# MODEL.WEIGHTS ./output/model_best_s_110000_nkc.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \

# # visualize the prediction json file
# python ./tools/visualize_json_results_rpn.py \
# --input ./output/inference/coco_instances_results.json \
# --input-gt ./datasets/humanware/annotations/instances_test_custom.json \
# --output ./output/custom_rpn_fp_07_12k_2k \
# --dataset humanware_test_custom \
# --conf-threshold 0 \
# --show-unique-boxes \
# --max-boxes 10000 \
# --small-region-px 0 \
# --score-t 0.5 \
# --iou-t 0.5 \
#make sure to change --input-gt and --dataset to relavive dataset in config

# # # visualize detection results from finetuned detectors on hw images

# #################################################################################
# # RN50, Finetune (COCO) on Humanware
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco_rpn.yaml \
# MODEL.WEIGHTS ./output/model_best_s_110000_nkc.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \

# # visualize the prediction json file
# python ./tools/visualize_json_results_rpn.py \
# --input ./output/inference/coco_instances_results.json \
# --input-gt ./datasets/humanware/annotations/instances_test_collected.json \
# --output ./output/hw_rpn_fp_09_24k_4k \
# --dataset humanware_test_collected \
# --conf-threshold 0 \
# --show-unique-boxes \
# --max-boxes 10000 \
# --small-region-px 0 \
# --no-vis \
# --score-t 0.5 \
# --iou-t 0.5 \


# # # # visualize detection results from finetuned detectors on basicAI images

# # #################################################################################
# # RN50, Finetune (COCO) on basicAI
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco_rpn.yaml \
# MODEL.WEIGHTS ./output/model_best_s_110000_nkc.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \

# # visualize the prediction json file
# python ./tools/visualize_json_results_rpn.py \
# --input ./output/inference/coco_instances_results.json \
# --input-gt ./datasets/humanware/annotations/instances_test_basic.json \
# --output ./output/basic_rpn_fp_09_6k_1k \
# --dataset humanware_test_basic \
# --conf-threshold 0 \
# --show-unique-boxes \
# --max-boxes 10000 \
# --small-region-px 0 \
# --score-t 0.5 \
# --iou-t 0.5 \


# # # visualize detection results from finetuned detectors on Awakening images

# #################################################################################
# RN50, Finetune (COCO) on basicAI
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco_rpn.yaml \
MODEL.WEIGHTS ./output/model_best_s_110000_nkc.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \

# visualize the prediction json file
python ./tools/visualize_json_results_rpn.py \
--input ./output/inference/coco_instances_results.json \
--input-gt ./datasets/humanware/annotations/instances_test_awake.json \
--output ./output/awake_rpn_fp_07_24k_4k \
--dataset humanware_test_awake \
--conf-threshold 0 \
--show-unique-boxes \
--max-boxes 10000 \
--small-region-px 0 \
--no-vis \
--score-t 0.5 \
--iou-t 0.5 \

