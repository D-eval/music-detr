# -*- coding: utf-8 -*-
"""配置目录入口。

提供对 config 动态加载的支持。
"""

from .config_uni import add_ddetrsvluni_config

try:
    from ..detectron22.config import get_cfg
except ImportError:
    raise ImportError('detectron22 is required for config generation. Please add the parent Open-Det2 path to PYTHONPATH.')


def get_default_config():
    cfg = get_cfg()
    add_ddetrsvluni_config(cfg)
    return cfg

__all__ = ['get_default_config', 'add_ddetrsvluni_config']
