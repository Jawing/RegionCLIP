from flask import Flask, request, Response, jsonify
import numpy as np
import cv2
from subprocess import call
import json
import base64
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
config_dir = './server_config.pkl'
classes = []
cat_file = './classes.txt'
with open(cat_file, 'r') as f:
    for line in f:
        classes.append(line.strip())

pred = None #prediction class stores model settings
cfg = None
from tools.train_net import model_inference_img
from tools.train_net import setup_model_cfg_pred
import pickle
#inital config setup
call(['bash',config_script])
with open(config_dir, 'rb') as file:
    cfg = pickle.load(file)
pred = setup_model_cfg_pred(cfg)

#pre load inference cache (speed up first inference by 1 second)
inference_img = './datasets/custom_images/test.jpg'
img = cv2.imread(inference_img,cv2.IMREAD_COLOR)
_ = model_inference_img(pred,img)

#set threshold for config
def set_thresholds(cfg,iou_threshold=0.2, conf_threshold=0.6):
    cfg['MODEL']['ROI_HEADS']['NMS_THRESH_TEST'] = iou_threshold
    cfg['MODEL']['ROI_HEADS']['SCORE_THRESH_TEST'] = conf_threshold
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

    global cfg,pred
    cfg = set_thresholds(cfg,iou_threshold=iou, conf_threshold=conf)
    pred = setup_model_cfg_pred(cfg)

# route http posts to this method
@app.route('/api/detection', methods=['POST'])
def detect_objects():
    """run inference model and output json bboxes as the response"""
    #get total time after turning off all print/time functions
    start = time.time()
    r = request.json
    image_content = r['image']
    nparr = np.frombuffer(base64.b64decode(image_content), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


    start_inf = time.time()
    #get inference
    result = model_inference_img(pred,img)
    end_inf = time.time()
    logging.info(f'Inference time: {end_inf - start_inf}') #0.5 seconds #(first run always 1.1 second slower)

    #compute prediction and output real json in response
    jsonResponse = add_category_name(result)
    end = time.time()
    
    logging.info(f'Number of detections: {len(jsonResponse)}')
    logging.info(f'Total detection time: {end - start}') #2 seconds
    jsonResponse = json.dumps(jsonResponse)
    logging.info(jsonResponse)

    response = Response(response=jsonResponse, status=200, mimetype="application/json")
    #response.headers.add('content-length', len(jsonResponse))
    return response

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(port=5200, debug=True, threaded=True)
    #from android studio emulator
    #app.run(host="172.31.6.26", debug=True, threaded=True)