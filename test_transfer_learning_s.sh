# evaluate our trained open-vocabulary object detectors, {RN50, RN50x4} x {COCO, LVIS}

#RN50, HUMANWARE (COCO) finetuned humanware
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_coco_eval.yaml \
MODEL.WEIGHTS ./output/model_best_small_110000.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
# MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \


# RN50, HUMANWARE (LVIS) finetuned humanware
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_lvis_eval.yaml \
# MODEL.WEIGHTS ./output/model_final_lvis_18000.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
# MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \
# MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \



