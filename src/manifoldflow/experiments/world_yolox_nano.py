"""YOLOX-Nano experiment for compatible tabletop COCO splits."""

from __future__ import annotations

import os
from pathlib import Path

import torch.nn as nn
from yolox.exp import Exp as YoloXExp


class Exp(YoloXExp):
    def __init__(self) -> None:
        super().__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.5
        self.enable_mixup = False
        self.num_classes = 4
        self.data_dir = os.environ.get(
            "MANIFOLDFLOW_WORLD_DATASET", "artifacts/world-sim"
        )
        Path(self.data_dir, "annotations").mkdir(parents=True, exist_ok=True)
        self.train_ann = "../train/coco.json"
        self.val_ann = "../validation/coco.json"
        self.max_epoch = 40
        self.warmup_epochs = 1
        self.no_aug_epochs = 5
        self.eval_interval = 2
        self.data_num_workers = 4
        self.exp_name = "world_yolox_nano"

    def get_model(self, sublinear: bool = False):
        del sublinear

        def initialize(module):
            for layer in module.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eps = 1e-3
                    layer.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

            channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=channels,
                act=self.act,
                depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=channels,
                act=self.act,
                depthwise=True,
            )
            self.model = YOLOX(backbone, head)
        self.model.apply(initialize)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train/images",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="validation/images",
            img_size=self.test_size,
            preproc=ValTransform(legacy=bool(kwargs.get("legacy", False))),
        )
