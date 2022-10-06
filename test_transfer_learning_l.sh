# evaluate our trained open-vocabulary object detectors, {RN50, RN50x4} x {COCO, LVIS}

# RN50x4, HUMANWARE (COCO) finetuned humanware
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_coco_eval.yaml \
MODEL.WEIGHTS ./models/model_best_large_90000.pth \
MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50x4_nn5.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50x4_nn5.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
# MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \

# # RN50x4, HUMANWARE (LVIS) finetuned lvis
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_lvis_eval.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-lvis_rn50x4 \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50x4_nn5.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50x4_nn5.pth \
# MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \
# MODEL.CLIP.TEXT_EMB_DIM 640 \
# MODEL.RESNETS.DEPTH 200 \
# MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
# MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION 18 \
# MODEL.RESNETS.RES2_OUT_CHANNELS 320 \
# MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \




