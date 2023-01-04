# visualize detection results from finetuned detectors on custom images

#################################################################################

# # RN50, HUMANWARE (COCO) on custom
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_coco.yaml \
# MODEL.WEIGHTS ./output/model_best_s_110000_nkc.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \
# # MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \

#RN50, HUMANWARE (COCO) finetuned humanware 
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_coco_eval.yaml \
MODEL.WEIGHTS ./models/ model_best_s_27500_basic.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_80.pth \
MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \

# visualize the prediction json file 
python ./tools/visualize_json_results.py \
--input ./output/inference/coco_instances_results.json \
--output ./output/basic_vis \
--dataset humanware_test_basic \
--conf-threshold 0.25 \
--show-unique-boxes \
--max-boxes 10000 \
--small-region-px 0 \

#MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/
#concept_embeds_basic63_coco_86.pth
#concept_embeds_basic_68.pth
#concept_embeds_basic_coco.pth
#concept_embeds_rclip_rn50_nn5.pth
#concept_embeds_c.pth

#MODEL.WEIGHTS ./output/ or ./models/
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
#model_best_s_77500_nkc_fbut
#model_best_s_25000_nkc_ubutfine
#model_best_s_110000_cup_pot.pth
#model_best_s_110000_nkc.pth

# # visualize the prediction json file for lvis custom
# python ./tools/visualize_json_results.py \
# --input ./output/inference/lvis_instances_results.json \
# --output ./output/zs_hw_pretrained_s \
# --dataset humanware_test_custom_img \
# --conf-threshold 0.25 \
# --show-unique-boxes \
# --max-boxes 10000 \
# --small-region-px 0 \

#change ood folder images
#rsync -a --delete custom_images_hw_test/ custom_images/
#rsync -a --delete custom_images_basic_test/ custom_images/


# # RN50, HUMANWARE (LVIS) on custom 
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_lvis.yaml \
# MODEL.WEIGHTS ./output/model_final_lvis_18000.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5key.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5key.pth \
# MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \

# visualize the prediction json file
# python ./tools/visualize_json_results.py \
# --input ./output/inference/lvis_instances_results.json \
# --output ./output/custom_lvis_s \
# --dataset humanware_val_custom_img \
# --conf-threshold 0.10 \
# --show-unique-boxes \
# --max-boxes 100 \
# --small-region-px 420 \
#min detected box area 211