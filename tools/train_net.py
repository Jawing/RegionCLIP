#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
#disable all warnings and loggings
import warnings
warnings.filterwarnings("ignore")
# logging.disable()
import time
from torch import nn
from contextlib import ExitStack
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_context
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
class Predictor(DefaultPredictor):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)


    @classmethod
    def test(cls, cfg, model, evaluators=None):
            dataset_name = 'lvis_v1_val_custom_img'
            data_loader = cls.build_test_loader(cfg, dataset_name)
            evaluator = cls.build_evaluator(cfg, dataset_name)

            #inside inference_on_dataset function
            evaluator.reset()
            with ExitStack() as stack:
                if isinstance(model, nn.Module):
                    stack.enter_context(inference_context(model))
                stack.enter_context(torch.no_grad())
                for idx, inputs in enumerate(data_loader):
                    outputs = model(inputs)
                    evaluator.process(inputs, outputs)

            #for storing in ./output/inference/lvis_instances_results.json
            #_ = evaluator.evaluate()

            return evaluator._predictions[0]['instances']
    @classmethod
    def process(self, output):
        """
        Args:
            inputs: the inputs to a LVIS model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a LVIS model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        instances = output["instances"].to("cpu")
        return instances_to_coco_json(instances, 0)
    
    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            
            return Predictor.process(predictions)
            
class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args) #0.7second save
    return cfg

def setup_model_cfg(cfg):
    model = Trainer.build_model(cfg) #1.93 seconds (2.15 seconds cpu time)     
    #important checkpointer
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    ) #0.24 seconds
    if cfg.MODEL.META_ARCHITECTURE in ['CLIPRCNN', 'CLIPFastRCNN', 'PretrainFastRCNN'] \
        and cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None\
        and cfg.MODEL.CLIP.CROP_REGION_TYPE == 'RPN': # load 2nd pretrained model
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, bb_rpn_weights=True).resume_or_load(
            cfg.MODEL.CLIP.BB_RPN_WEIGHTS, resume=False
        )
    return model
def setup_model_cfg_pred(cfg):
    pred = Predictor(cfg)
    return pred
def model_inference_img(pred,img):
    result = pred(img)
    torch.cuda.empty_cache() # 0.02 seconds
    return result
def model_inference(cfg,model):
    #Trainer.test(cfg, model)
    #result doesn't have to be returned test function save in lvis_instances_results.json
    result = Predictor.test(cfg,model)
    torch.cuda.empty_cache() # 0.02 seconds
    pass


def main(args):
    cfg = setup(args) #0.03 seconds
    if args.eval_only:
        model = Trainer.build_model(cfg) #1.93 seconds (2.15 seconds cpu time)     
        #important checkpointer
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        ) #0.24 seconds
        if cfg.MODEL.META_ARCHITECTURE in ['CLIPRCNN', 'CLIPFastRCNN', 'PretrainFastRCNN'] \
            and cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None\
            and cfg.MODEL.CLIP.CROP_REGION_TYPE == 'RPN': # load 2nd pretrained model
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, bb_rpn_weights=True).resume_or_load(
                cfg.MODEL.CLIP.BB_RPN_WEIGHTS, resume=False
            )
        #use time.time() walltime time.process_time() cpu time()
        # res only returned for evaluations with ground truth
        res = Trainer.test(cfg, model) #4.5 seconds (2.43 seconds cpu time)

        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()

import copy

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    #print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
