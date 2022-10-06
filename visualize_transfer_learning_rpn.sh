# visualize detection results from finetuned detectors on custom images

#################################################################################
# RN50, Finetune (COCO) on custom
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco_rpn.yaml \
MODEL.WEIGHTS ./output/model_best_small_110000.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \

# visualize the prediction json file
python ./tools/visualize_json_results_rpn.py \
--input ./output/inference/lvis_instances_results.json \
--input-gt ./datasets/humanware/annotations/instances_test_custom.json \
--output ./output/custom_rpn \
--dataset humanware_test_custom_img \
--conf-threshold 0 \
--show-unique-boxes \
--max-boxes 10000 \
--small-region-px 0 \

