# train open-vocabulary object detectors (initialized by our pretrained RegionCLIP), {RN50, RN50x4} x {COCO, LVIS}

# RN50, HUMANWARE (Finetuned LVIS)
# python ./tools/train_net.py \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_lvis.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-lvis_rn50.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_P_RN50_NN_local.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_P_RN50_NN_local.pth \
# MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \

# cp ./output/model_final.pth ./output/model_final_lvis_9000_lvis_2im.pth



# # RN50x4, HUMANWARE (Finetuned LVIS)
# python ./tools/train_net.py \
# --num-gpus 2 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_lvis.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-lvis-fully1203_rn50x4.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50x4_nn5.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50x4_nn5.pth \
# MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED False \
# MODEL.CLIP.TEXT_EMB_DIM 640 \
# MODEL.RESNETS.DEPTH 200 \
# MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
# MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION 18 \
# MODEL.RESNETS.RES2_OUT_CHANNELS 320 \

#./pretrained_ckpt/rpn/rpn_lvis_1203_clipbackbone.pth

# RN50, HUMANWARE (Finetuned COCO)
# python ./tools/train_net.py \
# --num-gpus 2 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_coco.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-coco_rn50.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \

# cp ./output/model_final.pth ./output/model_final_coco_9000_coco_2im.pth

# RN50x4, HUMANWARE (Finetuned COCO)
python3 ./tools/train_net.py \
--num-gpus 2 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_coco.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-coco_rn50x4.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50x4_nn5.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50x4_nn5.pth \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \


#./pretrained_ckpt/rpn/rpn_coco_80.pth



#./output/concept_feats/concept_embeds_vild_rn50.pth
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_vild_rn50.pth
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_vild_rn50.pth

# # RN50, HUMANWARE (LVIS) #'gt_masks' error it's a mask dataset #fixed by MODEL, MASK_ON settings
# python3 ./tools/train_net.py \
# --resume \
# --num-gpus 2 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_lvis.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_P_RN50.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/Humanware_indoor6_J_P_RN50.pth \
# MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \

# RN50, COCO
# python3 ./tools/train_net.py \
# --num-gpus 1 \
# --config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_48_base_cls_emb.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \

# # RN50, LVIS
# python3 ./tools/train_net.py \
# --num-gpus 1 \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_866_base_cls_emb.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb.pth \
# MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \



# # RN50x4, COCO
# python3 ./tools/train_net.py \
# --num-gpus 1 \
# --config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_48_base_cls_emb_rn50x4.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb_rn50x4.pth \
# MODEL.CLIP.TEXT_EMB_DIM 640 \
# MODEL.RESNETS.DEPTH 200 \
# MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \

# # RN50x4, LVIS
# python3 ./tools/train_net.py \
# --num-gpus 1 \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_866_base_cls_emb_rn50x4.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb_rn50x4.pth \
# MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \
# MODEL.CLIP.TEXT_EMB_DIM 640 \
# MODEL.RESNETS.DEPTH 200 \
# MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
# MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION 18 \
# MODEL.RESNETS.RES2_OUT_CHANNELS 320 \