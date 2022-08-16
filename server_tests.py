import time
from subprocess import call
inst_file = './output/inference/lvis_instances_results.json'
model_config_file = './configs/HUMANWARE-InstanceDetection/server_config.yaml'
import json
import yaml
import cv2

from tools.train_net import model_inference, model_inference_img
from tools.train_net import setup_model_cfg, setup_model_cfg_pred
#config setup
import pickle
model = None #TODO
cfg = None #TODO
config_script = './config_init.sh'
call(['bash',config_script])
config_dir = './server_config.pkl'
with open(config_dir, 'rb') as file:
    cfg = pickle.load(file)
model = setup_model_cfg_pred(cfg)

#test image read write
inference_img = './datasets/custom_images/test.jpg'
img = cv2.imread(inference_img,cv2.IMREAD_COLOR)
#cv2.imwrite('./img.jpg', img)
print(type(img))
#get classes
classes = []
cls_file = './classes.txt'
with open(cls_file, 'r') as f:
    for line in f:
        classes.append(line.strip())
#add names to inst_file
def add_category_name(annos):
    for anno in annos:
        anno["category_name"] = classes[anno["category_id"]-1]
    return annos
#set threshold for yaml config file
def set_thresholds(cfg,iou_threshold=0.2, conf_threshold=0.6):
    # with open(config_file, "r") as ymlfile:
    #     cfg = yaml.safe_load(ymlfile)
    cfg['MODEL']['ROI_HEADS']['NMS_THRESH_TEST'] = iou_threshold
    cfg['MODEL']['ROI_HEADS']['SCORE_THRESH_TEST'] = conf_threshold
    # with open(config_file, "w") as ymlfile:
    #     yaml.safe_dump(cfg,ymlfile,default_flow_style=False)
    return cfg
    
start_inf = time.time()
cfg = set_thresholds(cfg,iou_threshold=0.3, conf_threshold=0.6)
#resetup model configs with new parameters
model = setup_model_cfg_pred(cfg)
#get inference (can be slow)

model_inference(cfg,model)
# model_inference_img(img)

with open(inst_file, 'r') as f:
    annos = json.load(f)

#compute prediction and output real json in response
jsonResponse = add_category_name(annos)

end_inf = time.time()
print(f'Number of detections: {len(jsonResponse)}')
print(json.dumps(jsonResponse))
print(f'Inference time: {end_inf - start_inf}')