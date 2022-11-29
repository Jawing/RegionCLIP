# train open-vocabulary object detectors (initialized by our pretrained RegionCLIP), {RN50, RN50x4} x {COCO, LVIS}

# RN50, HUMANWARE (Finetuned COCO)
python3 ./tools/train_net.py \
--num-gpus 2 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_coco_UniCL_Swin.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/unicl/swin_b_klite.pt \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
#MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
#MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \

#MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
#MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_basic_coco.pth \

#model_best_s_27500_basic.pth
#MODEL.WEIGHTS ./output/model_best_s_110000_nkc.pth \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-coco_rn50.pth \

