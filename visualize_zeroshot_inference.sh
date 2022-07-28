# visualize zero-shot inference results on custom images
########################################################

# Humanware RegionCLIP (RN50) COCO RPN ONLY
python3 ./tools/train_net.py \
--eval-only \
--num-gpus 1 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_P_RN50.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \

# visualize the prediction json file
python ./tools/visualize_json_results.py \
--input ./output/inference/lvis_instances_results.json \
--output ./output/custom_zs_iid_coco \
--dataset humanware_val_custom_img \
--conf-threshold 0.7 \
--show-unique-boxes \
--max-boxes 50 \
--small-region-px 210\

# Humanware RegionCLIP (RN50) LVIS RPN ONLY
python3 ./tools/train_net.py \
--eval-only \
--num-gpus 1 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_lvis.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_P_RN50.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \


# visualize the prediction json file
python ./tools/visualize_json_results.py \
--input ./output/inference/lvis_instances_results.json \
--output ./output/custom_zs_iid_lvis \
--dataset humanware_val_custom_img \
--conf-threshold 0.7 \
--show-unique-boxes \
--max-boxes 50 \
--small-region-px 210\

########################################################

# Humanware RegionCLIP (RN50x4)
# python3 ./tools/train_net.py \
# --eval-only \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_RN50x4.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.TEXT_EMB_DIM 640 \
# MODEL.RESNETS.DEPTH 200 \
# MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \

#output lvis_instances_results.json name can be changed in /home/wanjiz/RegionCLIP/detectron2/evaluation/lvis_evaluation.py line 150

# visualize the prediction json file
# python ./tools/visualize_json_results.py \
# --input ./output/inference/lvis_instances_results.json \
# --output ./output/custom_zs \
# --dataset humanware_val_custom_img \
# --conf-threshold 0.90 \
# --show-unique-boxes \
# --max-boxes 25 \
# --small-region-px 8100 \
# min detected box area 211
# TODO: assume max boxes are already sorted by score in lvis_instances_results.json but not TRUE

########################################################

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





