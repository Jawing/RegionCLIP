# Extract concept features for a list of concepts

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# RN50 concept embeddings HUMANWARE
python3 ${SCRIPT_DIR?}/tools/extract_concept_features.py \
--config-file ${SCRIPT_DIR?}/configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_lvis.yaml \
MODEL.WEIGHTS ${SCRIPT_DIR?}/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ${SCRIPT_DIR?}/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
INPUT_DIR ${SCRIPT_DIR?}/datasets/custom_concepts \
OUTPUT_DIR ${SCRIPT_DIR?}/output/concept_feats \
MODEL.CLIP.GET_CONCEPT_EMB True \

# # RN50x4 concept embeddings HUMANWARE
# python3 ./tools/extract_concept_features.py \
# --config-file ./configs/HUMANWARE-InstanceDetection/CLIP_fast_rcnn_R_50_C4_custom_img_lvis.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
# MODEL.CLIP.TEXT_EMB_DIM 640 \
# MODEL.RESNETS.DEPTH 200 \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# INPUT_DIR ./datasets/custom_concepts \
# OUTPUT_DIR ./output/concept_feats \
# MODEL.CLIP.GET_CONCEPT_EMB True \

# RN50 concept embeddings
# python3 ./tools/extract_concept_features.py \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# INPUT_DIR ./datasets/custom_concepts \
# OUTPUT_DIR ./output/concept_feats \
# MODEL.CLIP.GET_CONCEPT_EMB True \

# RN50x4 concept embeddings
# python3 ./tools/extract_concept_features.py \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
# MODEL.CLIP.TEXT_EMB_DIM 640 \
# MODEL.RESNETS.DEPTH 200 \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# INPUT_DIR ./datasets/custom_concepts \
# OUTPUT_DIR ./output/concept_feats \
# MODEL.CLIP.GET_CONCEPT_EMB True \
