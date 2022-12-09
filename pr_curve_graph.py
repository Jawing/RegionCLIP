from podm import coco_decoder
from podm.metrics import get_pascal_voc_metrics, MetricPerClass, get_bounding_boxes
from podm.visualize import plot_precision_recall_curve, plot_precision_recall_curve_all, plot_precision_recall_curve_comb
from podm.metrics import MethodAveragePrecision
import numpy as np
import os
import argparse


#define percentage thresholds
cf_percents = [0.99,0.97,0.95,0.93,0.9,0.85,0.8,0.75,0.7,0.5,0.25]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('gt_dataset_name', help='input annotated directory')
    parser.add_argument('instances_results', help='input instances results')
    parser.add_argument('output_dir', help='output directory of the labelme annotation files')
    args = parser.parse_args()


    save_fig_loc = args.output_dir
    #create save_fig_loc if not exist
    if not os.path.exists(save_fig_loc):
        os.makedirs(save_fig_loc)

    result_coco_json = args.instances_results

    gt_dataset_json=f'./datasets/humanware/annotations/instances_{args.gt_dataset_name}.json'
    classes = []

    #load datasets
    with open(gt_dataset_json) as fp:
            gt_dataset = coco_decoder.load_true_object_detection_dataset(fp)

    with open(result_coco_json) as fp:
        pred_dataset = coco_decoder.load_pred_object_detection_dataset(fp, gt_dataset)

    #get precision/recall/means
    gt_BoundingBoxes = get_bounding_boxes(gt_dataset)
    pd_BoundingBoxes = get_bounding_boxes(pred_dataset)
    results50,lrs_list_all = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, 0.5, 
                        method= MethodAveragePrecision.AllPointsInterpolation,
                        pr_percents= cf_percents)

    #get all mAP(0.5,0.05,0.95) and mean
    iou_list = np.arange(0.5,1.0,0.05)
    mAP_results_list = []
    for iou_thresh in iou_list:
        results, _ = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, iou_thresh,
                         method= MethodAveragePrecision.AllPointsInterpolation,
                         pr_percents= cf_percents)

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


    plot_precision_recall_curve_comb(results = results50,
                                dest_dir = f"{save_fig_loc}",
                                method = MethodAveragePrecision.AllPointsInterpolation,
                                show_interpolated_precision=False)
    #plot for AP50
    plot_precision_recall_curve_all(results = results50,
                                dest_dir = f"{save_fig_loc}",
                                method = MethodAveragePrecision.AllPointsInterpolation,
                                show_ap=True,
                                show_interpolated_precision=True,
                                lrs_list = lrs_list_all)

