from flask import Flask, request, Response, jsonify
import numpy as np
import cv2
from subprocess import call
import json
import base64
import yaml
import os
#time testing
import time

#logging
import logging
LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

#server file assume in RegionCLIP folder
app = Flask(__name__)

#default locations
config_script = './config_init.sh'
inst_file = './output/inference/lvis_instances_results.json'
model_config_file = './configs/HUMANWARE-InstanceDetection/server_config.yaml'
inference_img = './datasets/custom_images/test.jpg'
config_dir = './server_config.pkl'
classes = []

model = None
cfg = None
from tools.train_net import model_inference #imports bug the server #TODO
from tools.train_net import setup_model_cfg
import pickle
#inital config setup
call(['bash',config_script])
with open(config_dir, 'rb') as file:
    cfg = pickle.load(file)
model = setup_model_cfg(cfg)



#set threshold for yaml config file
def set_thresholds(cfg,iou_threshold=0.2, conf_threshold=0.6):
    # with open(config_file, "r") as ymlfile:
    #     cfg = yaml.safe_load(ymlfile)
    cfg['MODEL']['ROI_HEADS']['NMS_THRESH_TEST'] = iou_threshold
    cfg['MODEL']['ROI_HEADS']['SCORE_THRESH_TEST'] = conf_threshold
    # with open(config_file, "w") as ymlfile:
    #     yaml.safe_dump(cfg,ymlfile,default_flow_style=False)
    return cfg

#add names to inst_file
def add_category_name(annos):
    for anno in annos:
        anno["category_name"] = classes[anno["category_id"]-1]
    return annos

# initializations
@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

    #setup class names
    global classes
    cat_file = './classes.txt'
    with open(cat_file, 'r') as f:
        for line in f:
            classes.append(line.strip())
    logging.info(f'Model Detecting Classes: {classes}')


    pass

@app.route("/api/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    # with open(LOG_FILE, 'r') as f:
    #     return render_template('content.html', text=f.read())
    response = {}
    with open(LOG_FILE) as f:
        for line in f:
            response[line] = line
    return jsonify(response)

# route get/set thresholds
@app.route('/api/detection/get_thresholds', methods=['GET'])
def get_thresholds():
    response = {}
    response['NMS_THRESH_TEST'] = cfg['MODEL']['ROI_HEADS']['NMS_THRESH_TEST']
    response['SCORE_THRESH_TEST'] = cfg['MODEL']['ROI_HEADS']['SCORE_THRESH_TEST']
    return jsonify(response)
@app.route('/api/detection/set_thresholds', methods=['PUT'])
def set_thresholds():
    r = request.json
    #set thresholds
    iou = float(r['iou_threshold'])
    conf = float(r['conf_threshold'])
    logging.info(f"Set iou threshold = {iou}")
    logging.info(f"Set conf threshold = {conf}")

    global cfg,model
    cfg = set_thresholds(cfg,iou_threshold=iou, conf_threshold=conf)
    model = setup_model_cfg(cfg)

# route http posts to this method
@app.route('/api/detection', methods=['POST'])
def detect_objects():
    """run inference model and output json bboxes as the response"""
    #get total time after turning off all print/time functions
    start = time.time()
    r = request.json

    # #set thresholds
    # iou = float(r['iou_threshold'])
    # conf = float(r['conf_threshold'])
    # logging.info(f"Set iou threshold = {iou}")
    # logging.info(f"Set conf threshold = {conf}")
    # set_thresholds(model_config_file,iou_threshold=iou, conf_threshold=conf)
    
    image_content = r['image']
    
    nparr = np.frombuffer(base64.b64decode(image_content), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #img = cv2.imread(inference_img,cv2.IMREAD_COLOR)

    cv2.imwrite(inference_img, img)

    start_inf = time.time()

    #get inference (6 seconds per image) #TODO reduce inference time by preload model
    model_inference(cfg,model)
    # call(['bash',inference_script])

    end_inf = time.time()
    logging.info(f'Inference time: {end_inf - start_inf}')

    with open(inst_file, 'r') as f:
        annos = json.load(f)

    #remove img at the end
    os.remove(inference_img)

    #compute prediction and output real json in response
    jsonResponse = add_category_name(annos)
    end = time.time()
    
    logging.info(f'Number of detections: {len(jsonResponse)}')
    logging.info(f'Total detection time: {end - start}')

    return Response(response=json.dumps(jsonResponse), status=200, mimetype="application/json")

if __name__ == '__main__':
    app.run(port=5200, debug=True, threaded=True)
    #from android studio emulator
    #app.run(host="172.31.6.26", debug=True, threaded=True)