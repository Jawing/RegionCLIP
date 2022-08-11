from podm import coco_decoder
from podm.metrics import get_pascal_voc_metrics, MetricPerClass, get_bounding_boxes
from podm.visualize import plot_precision_recall_curve, plot_precision_recall_curve_all
from podm.metrics import MethodAveragePrecision
import numpy as np

#define ground truth and inference locations

gt_hw_json = './datasets/humanware/annotations/instances_test_collected.json'
gt_basic_json = './datasets/humanware/annotations/instances_test_basic.json'
gt_basicVal_json = './datasets/humanware/annotations/instances_val_basic.json'
#result_lvis_json = './output/inference/lvis_instances_results.json'

#define output coco json and ground true below
result_coco_json = './output/inference/coco_instances_results_full_basic.json'
basic = True
val = False


if basic:
    if val:
        save_fig_loc = './output/inference/basicVal_pr50'
    else:
        save_fig_loc = './output/inference/basic_pr50'
else:
    save_fig_loc = './output/inference/hw_pr50'

classes = []

#load datasets
if basic:
    if val:
        with open(gt_basicVal_json) as fp:
            gt_dataset = coco_decoder.load_true_object_detection_dataset(fp)
    else: 
        with open(gt_basic_json) as fp:
            gt_dataset = coco_decoder.load_true_object_detection_dataset(fp)
else:
    with open(gt_hw_json) as fp:
        gt_dataset = coco_decoder.load_true_object_detection_dataset(fp)

with open(result_coco_json) as fp:
    pred_dataset = coco_decoder.load_pred_object_detection_dataset(fp, gt_dataset)

#get precision/recall/means
gt_BoundingBoxes = get_bounding_boxes(gt_dataset)
pd_BoundingBoxes = get_bounding_boxes(pred_dataset)
results50,lrs_list_all = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, 0.5, method= MethodAveragePrecision.AllPointsInterpolation)

#get all mAP(0.5,0.05,0.95) and mean
iou_list = np.arange(0.5,1.0,0.05)
mAP_results_list = []
for iou_thresh in iou_list:
    results, _ = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, iou_thresh, method= MethodAveragePrecision.AllPointsInterpolation)
    
    # #get results for each class
    # for cls, metric in results.items():
    #     print(cls)
    #     label = metric.label
    #     print('ap', metric.ap)
    #     # print('precision', metric.precision)
    #     # print('interpolated_recall', metric.interpolated_recall)
    #     # print('interpolated_precision', metric.interpolated_precision)
    #     # print('tp', metric.tp)
    #     # print('fp', metric.fp)
    #     print('num_groundtruth', metric.num_groundtruth)
    #     print('num_detection', metric.num_detection)
        
    mAP = MetricPerClass.mAP(results)
    print(f"mean AP{int(iou_thresh*100)}: {mAP}")
    mAP_results_list.append(mAP)

print(f"overall mAP: {np.mean(mAP_results_list)}")



#plot for AP50
plot_precision_recall_curve_all(results = results50,
                            dest_dir = f"{save_fig_loc}",
                            method = MethodAveragePrecision.AllPointsInterpolation,
                            show_ap=True,
                            show_interpolated_precision=True,
                            lrs_list = lrs_list_all)

