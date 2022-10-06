# evaluate zero-shot inference {RN50, RN50x4} x {COCO, LVIS} x {GT, RPN}

#GT = only ground truth classification head (CLIP) scores without RPN scores

# RN50, HUMANWARE (COCO) RPN
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_coco_eval_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
MODEL.CLIP.CROP_REGION_TYPE RPN \
MODEL.CLIP.MULTIPLY_RPN_SCORE True \

# RN50, HUMANWARE (LVIS) RPN
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_lvis_eval_zsinf.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \
# MODEL.CLIP.CROP_REGION_TYPE RPN \
# MODEL.CLIP.MULTIPLY_RPN_SCORE True \

# RN50, GT, COCO Humanware
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_coco_eval_zsinf.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \
# MODEL.CLIP.CROP_REGION_TYPE GT \
# MODEL.CLIP.MULTIPLY_RPN_SCORE False \

# RN50, GT, LVIS Humanware
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_lvis_eval_zsinf.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \
# MODEL.CLIP.CROP_REGION_TYPE GT \
# MODEL.CLIP.MULTIPLY_RPN_SCORE False \
# MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.0001 \
