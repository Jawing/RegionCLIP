# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .cityscapes_panoptic import register_all_cityscapes_panoptic
from .coco import load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import get_lvis_instances_meta, register_lvis_instances
from .pascal_voc import register_pascal_voc

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_minival_100": ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}
_PREDEFINED_SPLITS_COCO["coco_ovd"] = {
    "coco_2017_ovd_all_train": ("coco/train2017", "coco/annotations/ovd_ins_train2017_all.json"),
    "coco_2017_ovd_b_train": ("coco/train2017", "coco/annotations/ovd_ins_train2017_b.json"),
    "coco_2017_ovd_t_train": ("coco/train2017", "coco/annotations/ovd_ins_train2017_t.json"),
    "coco_2017_ovd_all_test": ("coco/val2017", "coco/annotations/ovd_ins_val2017_all.json"),
    "coco_2017_ovd_b_test": ("coco/val2017", "coco/annotations/ovd_ins_val2017_b.json"),
    "coco_2017_ovd_t_test": ("coco/val2017", "coco/annotations/ovd_ins_val2017_t.json"),
}


_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014_100.json",
    ),
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}

_PREDEFINED_SPLITS_COCO["humanware"] = {
    "humanware_train_full": ("humanware/train_full", "humanware/annotations/instances_train_full.json"),
    "humanware_train_awake": ("humanware/train_awake", "humanware/annotations/instances_train_awake.json"),
    "humanware_train_basic": ("humanware/train_basic", "humanware/annotations/instances_train_basic.json"),

    "humanware_val_full": ("humanware/val_full", "humanware/annotations/instances_val_full.json"),
    "humanware_val_awake": ("humanware/val_awake", "humanware/annotations/instances_val_awake.json"),
    "humanware_val_basic": ("humanware/val_basic", "humanware/annotations/instances_val_basic.json"),

    "humanware_test_full": ("humanware/test_full", "humanware/annotations/instances_test_full.json"),
    "humanware_test_awake": ("humanware/test_awake", "humanware/annotations/instances_test_awake.json"),
    "humanware_test_basic": ("humanware/test_basic", "humanware/annotations/instances_test_basic.json"),
    "humanware_test_collected": ("humanware/test_collected", "humanware/annotations/instances_test_collected.json"),
    "humanware_test_custom": ("humanware/test_custom", "humanware/annotations/instances_test_custom.json"),
    
}

_PREDEFINED_SPLITS_COCO["humanware_c"] = {
    #set with extra class labels added
    "humanware_train_full_c": ("humanware/train_full", "humanware/annotations/instances_train_full_c.json"),
    "humanware_train_awake_c": ("humanware/train_awake", "humanware/annotations/instances_train_awake_c.json"),
    "humanware_train_basic_c": ("humanware/train_basic", "humanware/annotations/instances_train_basic_c.json"),
    "humanware_train_basic_coco": ("humanware/train_basic_coco", "humanware/annotations/instances_train_basic_coco.json"),
    "humanware_train_basic_63": ("humanware/train_basic_63", "humanware/annotations/instances_train_basic_63.json"),
    "humanware_train_basic_63_all": ("humanware/train_basic_63", "humanware/annotations/instances_train_basic_63_all.json"),
    "humanware_train_basic_63_coco": ("humanware/train_basic_63_coco", "humanware/annotations/instances_train_basic_63_coco.json"),
    
    "humanware_val_full_c": ("humanware/val_full", "humanware/annotations/instances_val_full_c.json"),
    "humanware_val_awake_c": ("humanware/val_awake", "humanware/annotations/instances_val_awake_c.json"),
    "humanware_val_basic_c": ("humanware/val_basic", "humanware/annotations/instances_val_basic_c.json"),
    "humanware_val_basic_coco": ("humanware/val_basic_coco", "humanware/annotations/instances_val_basic_coco.json"),
    "humanware_val_basic_63": ("humanware/val_basic_63", "humanware/annotations/instances_val_basic_63.json"),
    "humanware_val_basic_63_all": ("humanware/val_basic_63", "humanware/annotations/instances_val_basic_63_all.json"),
    "humanware_val_basic_63_coco": ("humanware/val_basic_63_coco", "humanware/annotations/instances_val_basic_63_coco.json"),

    "humanware_test_full_c": ("humanware/test_full", "humanware/annotations/instances_test_full_c.json"),
    "humanware_test_awake_c": ("humanware/test_awake", "humanware/annotations/instances_test_awake_c.json"),
    "humanware_test_basic_c": ("humanware/test_basic", "humanware/annotations/instances_test_basic_c.json"),
    "humanware_test_collected_c": ("humanware/test_collected", "humanware/annotations/instances_test_collected_c.json"),
    "humanware_test_custom_c": ("humanware/test_custom", "humanware/annotations/instances_test_custom_c.json"),
    "humanware_test_basic_63": ("humanware/test_basic_63", "humanware/annotations/instances_test_basic_63.json"),
    "humanware_test_basic_63_all": ("humanware/test_basic_63", "humanware/annotations/instances_test_basic_63_all.json"),

    }

_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        if dataset_name == 'coco_ovd':  # for zero-shot split
            for key, (image_root, json_file) in splits_per_dataset.items():
                # Assume pre-defined datasets live in `./datasets`.
                register_coco_instances(
                    key,
                    {}, # empty metadata, it will be overwritten in load_coco_json() function
                    os.path.join(root, json_file) if "://" not in json_file else json_file,
                    os.path.join(root, image_root),
                )
        else: # default splits
            for key, (image_root, json_file) in splits_per_dataset.items():
                # Assume pre-defined datasets live in `./datasets`.
                register_coco_instances(
                    key,
                    _get_builtin_metadata(dataset_name),
                    os.path.join(root, json_file) if "://" not in json_file else json_file,
                    os.path.join(root, image_root),
                )

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "separated" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic FPN
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            _get_builtin_metadata("coco_panoptic_standard"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )

# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    # openset setting
    "lvis_v1": {
        "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    # custom image setting
    "lvis_v1_custom_img": {
        "lvis_v1_train_custom_img": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val_custom_img": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev_custom_img": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge_custom_img": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    #temp placeholder for custom image evaluations
    "humanware_custom_img": {
        "humanware_train_custom_img": ("humanware/train_full", "humanware/annotations/instances_train_full.json"),
        "humanware_val_custom_img": ("humanware/val_full", "humanware/annotations/instances_val_full.json"),
        "humanware_test_custom_img": ("humanware/test_custom", "humanware/annotations/instances_test_custom.json"),
    },

    # regular fully supervised setting
    "lvis_v1_fullysup": {
        "lvis_v1_train_fullysup": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val_fullysup": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev_fullysup": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge_fullysup": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            if dataset_name == "lvis_v1":
                args = {'filter_open_cls': True, 'run_custom_img': False}
            elif dataset_name == 'lvis_v1_custom_img':
                args = {'filter_open_cls': False, 'run_custom_img': True}
            elif dataset_name == 'lvis_v1_fullysup':
                args = {'filter_open_cls': False, 'run_custom_img': False}
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                args,
            )


# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}


def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_all_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )


# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_coco(_root)
    register_all_lvis(_root)
    register_all_cityscapes(_root)
    register_all_cityscapes_panoptic(_root)
    register_all_pascal_voc(_root)
    register_all_ade20k(_root)
