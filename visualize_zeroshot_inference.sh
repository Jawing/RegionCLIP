# visualize zero-shot inference results
########################################################

# Custom RegionCLIP (RN50) COCO/LVIS on custom folder
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-coco_rn50.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \

# #lvis settings
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_lvis_zsinf.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-lvis_rn50.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5key.pth \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
# MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \

#pretrained/finetune models
#regionclip_pretrained-cc_rn50.pth
#regionclip_finetuned-coco_rn50.pth
#regionclip_finetuned-lvis_rn50.pth

#note to do zsinf set MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH to a different embedding than original trained model embedding
#concept_embeds_rclip_rn50_nn5.pth
#googlecc_nouns_6250_emb.pth
#coco_48_base_cls_emb.pth
#lvis_866_base_cls_emb.pth
#lvis_1203_cls_emb.pth
#coco_65_cls_emb.pth
#coco_nouns_4764_emb.pth

# # visualize coco json file for dataset test images
# python ./tools/visualize_json_results.py \
# --input ./output/inference/coco_instances_results.json \
# --output ./output/zs_hw_finetune_coco_all \
# --dataset humanware_test_collected \
# --conf-threshold 0.25 \
# --show-unique-boxes \
# --max-boxes 10000 \
# --small-region-px 0 \


# # Pretrained RegionCLIP (RN50) COCO on custom folder LVIS settings
# python3 ./tools/train_net.py \
# --eval-only \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco_zsinf.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# #MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \

# # visualize the prediction json file
# python ./tools/visualize_json_results.py \
# --input ./output/inference/lvis_instances_results.json \
# --output ./output/custom_zsinf_coco_s \
# --dataset humanware_val_custom_img \
# --conf-threshold 0.7 \
# --show-unique-boxes \
# --max-boxes 50 \
# --small-region-px 210\

# # Humanware RegionCLIP (RN50) LVIS
# python3 ./tools/train_net.py \
# --eval-only \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_lvis_zsinf.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \


# # visualize the prediction json file
# python ./tools/visualize_json_results.py \
# --input ./output/inference/lvis_instances_results.json \
# --output ./output/custom_zsinf_lvis_s \
# --dataset humanware_val_custom_img \
# --conf-threshold 0.7 \
# --show-unique-boxes \
# --max-boxes 50 \
# --small-region-px 210\

########################################################

# Humanware RegionCLIP (RN50x4) LVIS
# python3 ./tools/train_net.py \
# --eval-only \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_lvis_zsinf.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50x4_nn5.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.TEXT_EMB_DIM 640 \
# MODEL.RESNETS.DEPTH 200 \
# MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \

#output lvis_instances_results.json name can be changed in /home/wanjiz/RegionCLIP/detectron2/evaluation/lvis_evaluation.py line 150

# visualize the prediction json file
# python ./tools/visualize_json_results.py \
# --input ./output/inference/lvis_instances_results.json \
# --output ./output/custom_zsinf_lvis_l \
# --dataset humanware_val_custom_img \
# --conf-threshold 0.90 \
# --show-unique-boxes \
# --max-boxes 25 \
# --small-region-px 8100 \
# min detected box area 211
# TODO: assume max boxes are already sorted by score in lvis_instances_results.json but not TRUE


########################################################
#zeroshot pretrained settings with lvis 1203 categories

# RegionCLIP (RN50x4)
# python3 ./tools/train_net.py \
# --eval-only \
# --num-gpus 1 \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_custom_img.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb_rn50x4.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.TEXT_EMB_DIM 640 \
# MODEL.RESNETS.DEPTH 200 \
# MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \

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

# RegionCLIP (RN50)
# python3 ./tools/train_net.py \
# --eval-only \
# --num-gpus 1 \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_custom_img.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \

# # visualize the prediction json file
# python ./tools/visualize_json_results.py \
# --input ./output/inference/lvis_instances_results.json \
# --output ./output/regions \
# --dataset lvis_v1_val_custom_img \
# --conf-threshold 0.05 \
# --show-unique-boxes \
# --max-boxes 25 \
# --small-region-px 8100\


