#generate pr curves for each individual coco_instances_results.json file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

#customizable ground truth names and output directory
#valid_basic, see /data/Object_detection/data/Indoor_objectDetection/humanware/annotations for names
GT_DATASET_NAME='test_full' 
INSTANCES_RESULTS='./output/inference/coco_instances_results.json' #'./output/inference/lvis_instances_results.json'
OUT_DIR='./tools/vis_pr_curve/basic_coco_full'

printf "\nCurrent directory for script: ${SCRIPT_DIR?}"
printf "\nGenerating pr curves for ${GT_DATASET_NAME?} dataset\n"
python ${SCRIPT_DIR?}/pr_curve_graph.py ${GT_DATASET_NAME?} ${INSTANCES_RESULTS?} ${OUT_DIR?}