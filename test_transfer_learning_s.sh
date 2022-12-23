# evaluate our trained open-vocabulary object detectors, {RN50, RN50x4} x {COCO, LVIS}

#RN50, HUMANWARE (COCO) finetuned humanware
#make sure to run /data/Object_detection/data/Indoor_objectDetection/anno2coco_c_sync.sh
#with extra labels
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_coco_eval.yaml \
MODEL.WEIGHTS ./output/model_best_s_55000_basic63_LSJelevator.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
# MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \

#MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/
#concept_embeds_basic63_coco_86.pth
#concept_embeds_basic_68.pth
#concept_embeds_basic_coco.pth
#concept_embeds_rclip_rn50_nn5.pth
#concept_embeds_c.pth

#MODEL.WEIGHTS ./output/
#model_best_s_55000_basic63_LSJelevator.pth
#model_best_s_40000_basic63_LSJ.pth
#model_best_s_135000_basic63_coco_86.pth
#model_best_s_22000_basic63.pth
#model_best_s_132500_basic_coco.pth
#model_best_s_16000_basic_LSJ_ADAM.pth
#model_best_s_27000_basic_LSJ_SGD.pth
#model_best_s_90000_nkc_lsj.pth
#model_best_s_100_basic_boosted.pth
#model_best_s_27500_basic.pth
#model_best_s_50000_nkc_ubut.pth
#model_best_s_7500_nkc_fbutfine.pth
#model_best_s_77500_nkc_fbut
#model_best_s_25000_nkc_ubutfine
#model_best_s_110000_cup_pot.pth
#model_best_s_110000_nkc.pth

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



