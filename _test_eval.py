# -*- coding: utf-8 -*-
"""在项目根目录运行 `python3.9 test_model.py`。

这个脚本对照原项目的训练路径，构造一个最小可运行样本并打印 loss。
"""

import os
import sys

import torch


# 把项目根目录（my）以及上级目录（Open-Det2）加入模块搜索路径，方便直接执行
ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.abspath(os.path.join(ROOT, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from ddetrs_vl_uni_models import DDETRSVLUni, get_default_config
from ddetrs_vl_uni_models.detectron22.structures import Boxes, Instances
from ddetrs_vl_uni_models.models.data.custom_dataset_mapper import ObjDescription


device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = get_default_config()
cfg.MODEL.DEVICE = device
cfg.MODEL.TEXT.BEAM_SIZE = 1
cfg.INPUT.MIN_SIZE_TEST = 512
cfg.INPUT.MAX_SIZE_TEST = 512

model = DDETRSVLUni(cfg).to(device)
model.eval()


def build_dummy_batch(image_size=256):
    image = torch.randn(3, image_size, image_size, device=device)

    instances = Instances((image_size, image_size))
    instances.gt_boxes = Boxes(
        torch.tensor([[32.0, 40.0, 180.0, 200.0]], dtype=torch.float32, device=device)
    )
    instances.gt_classes = torch.tensor([0], dtype=torch.int64, device=device)
    instances.gt_object_descriptions = ObjDescription(["object"])

    return [
        {
            "image": image,
            "height": image_size,
            "width": image_size,
            "dataset_source": "coco",
            "anno_type": "box",
            "instances": instances,
        }
    ]


batched_inputs = build_dummy_batch(image_size=256)

output = model(batched_inputs)
