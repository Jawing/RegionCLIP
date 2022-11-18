#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A script for region feature extraction
"""

import os
import torch
from torch.nn import functional as F
import numpy as np
import time
import cv2
import copy

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.structures import Boxes, BoxMode

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.modeling.meta_arch.clip_rcnn import visualize_proposals

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def get_inputs(cfg, file_name):
    """ Given a file name, return a list of dictionary with each dict corresponding to an image
    (refer to detectron2/data/dataset_mapper.py)
    """
    # image loading
    dataset_dict = {}
    image = utils.read_image(file_name, format=cfg.INPUT.FORMAT)
    dataset_dict["height"], dataset_dict["width"] = image.shape[0], image.shape[1] # h, w before transforms
    
    # image transformation
    augs = utils.build_augmentation(cfg, False)
    augmentations = T.AugmentationList(augs) # [ResizeShortestEdge(short_edge_length=(800, 800), max_size=1333, sample_style='choice')]
    aug_input = T.AugInput(image)
    transforms = augmentations(aug_input)
    image = aug_input.image
    h, w = image.shape[:2]  # h, w after transforms
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

    return [dataset_dict]

def create_model(cfg):
    """ Given a config file, create a detector
    (refer to tools/train_net.py)
    """
    # create model
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    if cfg.MODEL.META_ARCHITECTURE in ['CLIPRCNN', 'CLIPFastRCNN', 'PretrainFastRCNN'] \
        and cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None\
        and cfg.MODEL.CLIP.CROP_REGION_TYPE == 'RPN': # load 2nd pretrained model
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, bb_rpn_weights=True).resume_or_load(
            cfg.MODEL.CLIP.BB_RPN_WEIGHTS, resume=False
        )
    
    #assert model.clip_crop_region_type == "RPN"
    assert model.use_clip_c4 # use C4 + resnet weights from CLIP
    assert model.use_clip_attpool # use att_pool from CLIP to match dimension
    model.roi_heads.box_predictor.vis = True # get confidence scores before multiplying RPN scores, if any
    for p in model.parameters(): p.requires_grad = False
    model.eval()
    return model

def extract_region_feats(cfg, model, batched_inputs, file_name):
    """ Given a model and the input images, extract region features and save detection outputs into a local file
    (refer to detectron2/modeling/meta_arch/clip_rcnn.py)
    """
    # model inference
    # 1. localization branch: offline modules to get the region proposals           
    # select from GT or RPN
    if cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN":
        images = model.offline_preprocess_image(batched_inputs)
        features = model.offline_backbone(images.tensor)
        proposals, _ = model.offline_proposal_generator(images, features, None)    
    elif cfg.MODEL.CLIP.CROP_REGION_TYPE == "GT":
        proposals = []
        for r_i, b_input in enumerate(batched_inputs): 
            this_gt = copy.deepcopy(b_input["instances"])  # Instance
            gt_classes = this_gt._fields['gt_classes'].to("cuda")
            gt_boxes = this_gt._fields['gt_boxes'].to("cuda")
            this_gt._fields = {'proposal_boxes': gt_boxes, 
            'objectness_logits': torch.ones(gt_boxes.tensor.size(0)).to("cuda"),
            'gt_classes': gt_classes}
            proposals.append(this_gt) 
    #visualize_proposals(batched_inputs, proposals, model.input_format) 

    # 2. recognition branch: get 2D feature maps using the backbone of recognition branch
    images = model.preprocess_image(batched_inputs)
    features = model.backbone(images.tensor)

    # 3. given the proposals, crop region features from 2D image features
    proposal_boxes = [x.proposal_boxes for x in proposals]
    if cfg.MODEL.CLIP.CROP_REGION_TYPE == "GT":
        gt_classes = [x.gt_classes for x in proposals]

    box_features = model.roi_heads._shared_roi_transform(
        [features[f] for f in model.roi_heads.in_features], proposal_boxes, model.backbone.layer4
    )
    att_feats = model.backbone.attnpool(box_features)  # region features

    if cfg.MODEL.CLIP.TEXT_EMB_PATH is None: # save features of RPN regions
        results = model._postprocess(proposals, batched_inputs) # re-scale boxes back to original image size

        # save RPN outputs into files
        im_id = 0 # single image
        pred_boxes = results[im_id]['instances'].get("proposal_boxes").tensor # RPN boxes, [#boxes, 4]
        region_feats = att_feats # region features, [#boxes, d]

        saved_dict = {}
        saved_dict['boxes'] = pred_boxes.cpu().tolist()
        saved_dict['feats'] = region_feats.cpu().tolist()
        if cfg.MODEL.CLIP.CROP_REGION_TYPE == "GT":
            saved_dict['gt_classes'] = gt_classes[im_id].cpu().tolist()
    else: # save features of detection regions (after per-class NMS)
        # 4. prediction head classifies the regions (optional)
        predictions = model.roi_heads.box_predictor(att_feats)  # predictions[0]: class logits; predictions[1]: box delta
        pred_instances, keep_indices = model.roi_heads.box_predictor.inference(predictions, proposals) # apply per-class NMS
        results = model._postprocess(pred_instances, batched_inputs) # re-scale boxes back to original image size

        # save detection outputs into files
        im_id = 0 # single image
        pred_boxes = results[im_id]['instances'].get("pred_boxes").tensor # boxes after per-class NMS, [#boxes, 4]
        pred_classes = results[im_id]['instances'].get("pred_classes")# class predictions after per-class NMS, [#boxes], class value in [0, C]
        #pred_probs = F.softmax(predictions[0], dim=-1)[keep_indices[im_id]] # class probabilities, [#boxes, #concepts+1], background is the index of C
        region_feats = att_feats[keep_indices[im_id]] # region features, [#boxes, d]
        # assert torch.all(results[0]['instances'].get("scores") == pred_probs[torch.arange(pred_probs.shape[0]).cuda(), pred_classes]) # scores
        
        saved_dict = {}

        #convert boxes to x, y, w, h
        saved_dict['scores'] = results[im_id]['instances'].get("scores").cpu().tolist()
        saved_dict['boxes'] = BoxMode.convert(pred_boxes.cpu().numpy(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS).tolist()
        saved_dict['classes'] = pred_classes.cpu().tolist()
        #saved_dict['probs'] = pred_probs.cpu().tolist()
        saved_dict['feats'] = region_feats.cpu().tolist()
        if cfg.MODEL.CLIP.CROP_REGION_TYPE == "GT":
            saved_dict['gt_classes'] = gt_classes[im_id].cpu().tolist()
        img_name = os.path.basename(file_name).split('/')[-1]

        saved_dict['image_name'] = [img_name for x in range(pred_boxes.size(0))]
        saved_dict['boxes_img'] = []
        for idx, bbox in enumerate(saved_dict['boxes']):
            img = cv2.imread('./' + file_name)
            #img_name = os.path.basename(file_name).split('.')[0]
            x, y , w, h = [int(x) for x in bbox]
            img_crop = img[y:y+h, x:x+w]
            #resize to constant to fit display in TSNE vis
            min_size = 128
            (ih, iw) = img_crop.shape[:2]
            if ih > iw:
                img_crop = image_resize(img_crop, width = min_size)
            else:
                img_crop = image_resize(img_crop, height = min_size)
            # img_dir = cfg.OUTPUT_DIR+"/crops"
            # if not os.path.exists(img_dir):
            #     os.makedirs(img_dir)
            #img_path = "{}/{}_bbox{}bd.jpg".format(img_dir, img_name, idx)
            #print(os.path.abspath(img_path))
            #cv2.imwrite(img_path,img_crop)
            saved_dict['boxes_img'].append(img_crop)
        
    
    return saved_dict
    
    
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        r = height / float(h)
        # calculate the ratio of the height and construct the
        # dimensions
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


# def main_old(args):
#     cfg = setup(args)

#     # create model
#     model = create_model(cfg)

#     # input images
#     image_files = [os.path.join(cfg.INPUT_DIR, x) for x in os.listdir(cfg.INPUT_DIR)]
    
#     # process each image
#     start = time.time()
#     for i, file_name in enumerate(image_files):
#         if i % 100 == 0:
#             print("Used {} seconds for 100 images.".format(time.time()-start))
#             start = time.time()
        
#         # get input images
#         batched_inputs = get_inputs(cfg, file_name)

#         # extract region features
#         with torch.no_grad():
#             extract_region_feats(cfg, model, batched_inputs, file_name)



#     print("done!")

# # def get_inputs(cfg,model):
# #     DefaultTrainer.test(cfg, model)
import pickle
def main(args):
    cfg = setup(args)
    # create model
    model = create_model(cfg)

    # select input images and process
    data_loader = DefaultTrainer.get_data(cfg, model)


    all_feats = {}
    all_feats['boxes'] = []
    all_feats['classes'] = []
    #all_feats['probs'] = []
    all_feats['feats'] = []
    all_feats['image_name'] = []
    all_feats['boxes_img'] = []
    all_feats['scores'] = []
    if cfg.MODEL.CLIP.CROP_REGION_TYPE == "GT":
        all_feats['gt_classes'] = []
    #saved_path = os.path.join(cfg.OUTPUT_DIR)+"/"+cfg.DATASETS.TEST[0]+"_all_feats.pth"
    #all_feats = torch.load(saved_path)
    saved_path_pkl = os.path.join(cfg.OUTPUT_DIR)+"/" \
                    +cfg.DATASETS.TEST[0]+"_all_feats_" \
                    +cfg.MODEL.CLIP.CROP_REGION_TYPE+'_' \
                    +str(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST).replace('.', '') \
                    +".pkl"
    # with open(saved_path_pkl, 'rb') as fp:
    #     all_feats = pickle.load(fp)

    start = time.time()
    for i, batched_inputs in enumerate(data_loader):
        if i % 100 == 0:
            print("Used {} seconds for 100 images.".format(time.time()-start))
            start = time.time()
        # extract region features
        with torch.no_grad():
            saved_dict = extract_region_feats(cfg, model, batched_inputs, batched_inputs[0]["file_name"])
        all_feats['scores'] += saved_dict['scores']
        all_feats['boxes'] += saved_dict['boxes']
        all_feats['classes'] += saved_dict['classes']
        if cfg.MODEL.CLIP.CROP_REGION_TYPE == "GT":
            all_feats['gt_classes'] += saved_dict['gt_classes']
        #all_feats['probs'] += saved_dict['probs']
        all_feats['feats'] += saved_dict['feats']
        all_feats['image_name'] += saved_dict['image_name']
        all_feats['boxes_img'] += saved_dict['boxes_img']

        #images when score threshold set too low can take up a lot of memory
        # if i == 30:
        #     break

    #torch.save(all_feats, saved_path)
    print("number of classes: ",len(all_feats['classes']))
    if cfg.MODEL.CLIP.CROP_REGION_TYPE == "GT":
        print("number of gt_classes: ",len(all_feats['gt_classes']))
        diff=0
        for a, b in zip(all_feats['gt_classes'], all_feats['classes']):
            if a != b:
                diff += 1
        print("number of diff class: ", diff)
    print("number of feats: ",len(all_feats['feats']))
    print("number of scores: ",len(all_feats['scores']))
    print("number of boxes: ",len(all_feats['boxes']))
    print("saved to: ", saved_path_pkl)
    with open(saved_path_pkl, "wb") as f:
        pickle.dump(all_feats, f)
    print("done!")

    
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
