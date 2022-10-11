#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.data.datasets.coco_zeroshot_categories import COCO_UNSEEN_CLS, COCO_SEEN_CLS, COCO_OVD_ALL_CLS
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
import matplotlib.colors as mplc
def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    all_scores = np.asarray([predictions[i]["all_scores"] for i in chosen])
    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels
    ret.all_scores = all_scores
    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret

#compare ground truth and proposal by iou
#output the proposals ID that overlap with ground truth
#keep tallie number of proposals for each ground truth that overlap and output
def compare_box_proposals(dataset_predictions, gt_json, thresholds=None, area="all", limit=None):

    gt_overlaps = []
    num_pos = 0

    for prediction_dict in dataset_predictions:
        predictions = prediction_dict["proposals"]

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = predictions.objectness_logits.sort(descending=True)[1]
        predictions = predictions[inds]

        ann_ids = COCO(gt_json).getAnnIds(imgIds=prediction_dict["image_id"])
        anno = COCO(gt_json).loadAnns(ann_ids)
        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in anno
            if obj["iscrowd"] == 0
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes)
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]

        overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = (
        torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--input-gt", required=True, help="JSON file of ground truth dataset")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--show-unique-boxes", action='store_true', help="if true, only show one prediction for each box")
    parser.add_argument("--max-boxes", default=30, type=int, help="the maximum number of boxes to visualize per image")
    parser.add_argument("--small-region-px", default=0.0, type=float, help="the boxes with very few pixels will not be visualized")
    args = parser.parse_args()

    logger = setup_logger()

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif any(x in args.dataset for x in ['lvis','humanware']):
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1
    #TODO define ID mapping for Humanware
    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)
    cnt = 0
    iou_thresholds=0.5
    #number of total bounding boxes
    num_pos = 0
    #number of bounding boxes with no overlap
    num_nol = 0
    no_overlap_names = []
    for dic in tqdm.tqdm(dicts):
        cnt += 1
        if cnt < 0: #100:
            continue
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])
        if 'ourclip' in args.input:
            basename = basename.split(".")[0] + "_ours.jpg"
        print(basename)
        if args.show_unique_boxes: 
            seen_box = []
            unique_box_pred = []
            for pred in pred_by_image[dic["image_id"]][:args.max_boxes]:  # assume boxes are already sorted by score
                if pred['bbox'][2] * pred['bbox'][3] < args.small_region_px: # filter the small boxes
                    continue
                if pred['bbox'] in seen_box: # ignore other predictions of this seen box
                    continue
                else:
                    seen_box.append(pred['bbox'])
                    unique_box_pred.append(pred)
            print("Got {} boxes".format(len(unique_box_pred)))
            predictions = create_instances(unique_box_pred, img.shape[:2])
        else:  # default visualization
            predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        

        #load groundtruth
        coco_gt=COCO(args.input_gt)
        ann_ids = COCO(args.input_gt).getAnnIds(imgIds=dic["image_id"])
        anno = COCO(args.input_gt).loadAnns(ann_ids)
        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in anno
            if obj["iscrowd"] == 0
        ]
        gt_classes = np.array([
            obj["category_id"]-1
            for obj in anno
            if obj["iscrowd"] == 0
        ])
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes)
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0 or len(predictions.pred_boxes) == 0:
            continue

        num_pos += len(gt_boxes)

        #calculate and filter iou overlaps over ground truth boxes
        overlaps_scores = pairwise_iou(predictions.pred_boxes, gt_boxes)
        #store percentage of bounding box overlaping
        pt_overlap = []
        overlaps = overlaps_scores > iou_thresholds
        overlaps_each_bbox = torch.sum(overlaps,dim = 0)
        no_overlap= False
        for i,v in enumerate(overlaps_each_bbox):
            print(f"GT bbox# {i} has {v} RPN bbox overlapping")
            if v == 0:
                num_nol += 1
                no_overlap= True
            pt_overlap.append(f"{v}/{overlaps_scores.size(0)}")
        #if no overlap print image name
        if no_overlap==True:
            print(f"No overlap in {dic['file_name']}")
            no_overlap_names.append(dic['file_name'])

        #get best confidence score from all overlaps
        #store the percentage of bounding box overlaping with gt class as top
        pt_top_gt = []

        best_idx = []
        best_gt_scores = []
        best_ious = []
        for i, b in enumerate(overlaps.T):
            #get idx of each overlap
            b_overlap_idx = b.nonzero().T.tolist()[0]
            b_all_scores = predictions.all_scores[b_overlap_idx][:,:-1] #select everything other than bg class
            b_scores, c_idx = torch.max(torch.from_numpy(b_all_scores),dim=1)
            #TODO can filter based on b_score for n_top_gt within score threshold
            #output number of top_gt class detected over all detected boxes overlap
            n_top_gt = c_idx.tolist().count(gt_classes[i])
            pt_top_gt.append(f"{n_top_gt}")
            if len(b_overlap_idx) == 0:
                continue
            b_score, b_idx = torch.max(b_scores, dim = 0)
            b_idx = b_overlap_idx[b_idx.item()]
            best_ious.append(overlaps_scores.T[i,b_idx].item())
            best_gt_scores.append(predictions.all_scores[b_idx,gt_classes[i]])
            best_idx.append(b_idx)
        best_all_scores = predictions.all_scores[best_idx]
        best_classes = predictions.pred_classes[best_idx]
        best_scores = predictions.scores[best_idx]
        best_boxes= predictions.pred_boxes.tensor[best_idx]
        #print(best_scores,best_idx)

        #get idx of all overlapping boxes
        overlaps = torch.any(overlaps,dim = 1)
        overlap_idx = overlaps.nonzero().T.tolist()[0]

        #get top ious of the filtered bounding boxes
        top_ious = []
        top_idx = []
        top_gt_scores = []
        top_all_scores=[]
        for i, b in enumerate(overlaps_scores.T):
            #b = b.view(-1,1)
            max_iou, max_idx = torch.max(b,dim=0)
            #filter the overlapping bounding boxes based on iou
            if max_iou > iou_thresholds:
                top_ious.append(max_iou.item())
                top_idx.append(max_idx.item())
                top_gt_scores.append(predictions.all_scores[max_idx.item(),gt_classes[i]])
                top_all_scores.append(predictions.all_scores[max_idx.item()])
        #print(top_ious,top_idx)
        top_classes = predictions.pred_classes[top_idx]
        top_scores = predictions.scores[top_idx]
        top_boxes= predictions.pred_boxes.tensor[top_idx]
        top_best_eq= []
        #print/save diff stats if top iou and best scores are the same
        for i,idx in enumerate(top_idx):
            if top_idx[i] == best_idx[i]:
                top_best_eq.append(True)
            else:
                top_best_eq.append(False)



        ############
        #define custom colors and lengths and shades
        #set colors for gt boxes
        colors_gt = [np.array(list(mplc.BASE_COLORS['w'])) for _ in range(len(anno))]
        #set colors for prediction boxes
        colors = [np.array(list(mplc.BASE_COLORS['m'])) for _ in range(len(unique_box_pred))]
        colors = np.array(colors)
        #add color for overlapping boxes
        colors[overlap_idx] = list(mplc.BASE_COLORS['c'])
        #add color for top overlapping box
        colors[top_idx] = list(mplc.BASE_COLORS['y'])
        #add color for best overlapping box
        colors[best_idx] = list(mplc.BASE_COLORS['y'])
        #combine
        colors = np.concatenate((colors_gt,colors),axis=0)

        #set length for gt boxes
        box_length_gt = [8.0 for _ in range(len(anno))]
        #set length for prediction boxes
        box_length = [1.0 for _ in range(len(unique_box_pred))]
        box_length = np.array(box_length)
        #set length for overlapping boxes
        box_length[overlap_idx] = 3.0
        #set length for top overlapping box
        box_length[top_idx] = 6.0
        #set length for best overlapping box
        box_length[best_idx] = 6.0
        #combine
        box_length = np.concatenate((box_length_gt,box_length),axis=0)


        #set transparency for gt boxes
        alpha_gt = [1.0 for _ in range(len(anno))]
        #set length for prediction boxes
        alpha = [0.3 for _ in range(len(unique_box_pred))]
        alpha = np.array(alpha)
        #set length for overlapping boxes
        alpha[overlap_idx] = 0.5
        #set length for top overlapping box
        alpha[top_idx] = 1.0
        #set length for best overlapping box
        alpha[best_idx] = 1.0
        #combine
        alpha = np.concatenate((alpha_gt,alpha),axis=0)

        #reshift idx with gt add
        top_idx = np.array(top_idx)
        top_idx +=len(gt_classes)
        best_idx = np.array(best_idx)
        best_idx +=len(gt_classes)

        #get ground truth boxes and append stats next to the class label
        gt_dict = {'gt_scores':np.ones_like(gt_classes,dtype=np.float32),'gt_boxes':gt_boxes,'gt_classes':gt_classes}
        top_dict = {'top_scores':top_scores,'top_boxes':top_boxes,'top_classes':top_classes,'top_idx':top_idx,
                'top_iou':top_ious,'top_gt_scores':top_gt_scores,'top_all_scores':top_all_scores,'top_best_eq': top_best_eq}
        best_dict = {'best_scores':best_scores,'best_boxes':best_boxes,'best_classes':best_classes,'best_idx':best_idx,
                'best_iou':best_ious,'best_gt_scores':best_gt_scores,'best_all_scores':best_all_scores}
        stats = {'pt_overlap':pt_overlap, 'pt_top_gt':pt_top_gt}

        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions_rpn(predictions, gt_dict, top_dict,best_dict, stats, colors = colors,inc_label=True,box_length = box_length,alpha=alpha).get_image()

        # doesn't draw gt
        # vis = Visualizer(img, metadata)
        # vis_gt = vis.draw_dataset_dict(dic).get_image()

        concat = vis_pred # np.concatenate((vis_pred, vis_gt), axis=1)
        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
    
    print(f"Total bounding boxes: {num_pos}")
    print(f"Bounding boxes with no overlap: {num_nol}")
    print(f"Not detected by RPN: {(num_nol/num_pos)*100:.2f}%")
    print(f"Images location without overlap:")
    print('\n'.join('{}'.format(k) for k in no_overlap_names))