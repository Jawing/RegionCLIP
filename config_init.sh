#################################################################################

# Run model inference on input images in (~/RegionCLIP/datasets/custom_images)
# inference output at ~/RegionCLIP/output/inference/lvis_instances_results.json

# uncomment/comment to select between model before running server.py

# for text embeddings with extra label classes, make sure to add/remove relevant classes from './labels_5class_c.txt' 
# with correct text embedding order before running server

# #RN50 model trained on new basicAI keychain labels
python3 ./tools/config_init.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/HUMANWARE-InstanceDetection/server_config.yaml \
MODEL.WEIGHTS ./models/model_best_s_110000_nkc.pth \
MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \

#MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/
#concept_embeds_basic63_coco_86.pth
#concept_embeds_basic_68.pth
#concept_embeds_basic_coco.pth
#concept_embeds_rclip_rn50_nn5.pth

#MODEL.WEIGHTS ./models/
#model_best_s_147500_nkc_rpn80.pth
#model_best_s_65000_nkc_pretrained.pth
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
#model_best_s_77500_nkc_fbut.pth
#model_best_s_25000_nkc_ubutfine.pth
#model_best_s_110000_cup_pot.pth
#model_best_s_110000_nkc.pth

# #RN50x4 model inference trained on old basicAI keychain labels
# python3 ./tools/config_init.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/server_config.yaml \
# MODEL.WEIGHTS ./models/model_best_large.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50x4_nn5key.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50x4_nn5key.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
# MODEL.CLIP.TEXT_EMB_DIM 640 \
# MODEL.RESNETS.DEPTH 200 \
# MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \