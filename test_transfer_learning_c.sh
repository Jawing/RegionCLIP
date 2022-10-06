# evaluate our trained open-vocabulary object detectors, {RN50, RN50x4} x {COCO, LVIS}

#RN50, HUMANWARE (COCO) finetuned humanware final
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_coco_eval_c.yaml \
MODEL.WEIGHTS ./output/model_best_small_110000.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \
# MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \

#note to do zsinf or openset test set MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH to a different embedding