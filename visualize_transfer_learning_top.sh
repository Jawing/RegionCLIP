# #RN50, HUMANWARE (COCO) finetuned humanware 
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_coco_eval.yaml \
# MODEL.WEIGHTS ./output/model_best_s_110000_nkc.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_rclip_rn50_nn5.pth \

# # visualize the prediction json file
# python ./tools/visualize_json_results_top.py \
# --input ./output/inference/coco_instances_results.json \
# --output ./output/custom_top \
# --dataset humanware_test_custom \
# --conf-threshold 0 \
# --show-unique-boxes \
# --max-boxes 10000 \
# --small-region-px 0 \


#RN50, HUMANWARE (COCO) finetuned humanware cups and pots
#make sure to run /data/Object_detection/data/Indoor_objectDetection/anno2coco_c_sync.sh
#with cup and pot label
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_Humanware_coco_eval_c.yaml \
MODEL.WEIGHTS ./output/model_best_s_110000_cup_pot.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
MODEL.CLIP.TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./output/concept_feats/concept_embeds_c.pth \

# visualize the prediction json file
python ./tools/visualize_json_results_top.py \
--input ./output/inference/coco_instances_results.json \
--output ./output/custom_top_cup_pot_03 \
--dataset humanware_test_custom_c \
--conf-threshold 0.1 \
--show-unique-boxes \
--max-boxes 10000 \
--small-region-px 0 \
