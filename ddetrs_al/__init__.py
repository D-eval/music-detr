"""ddetrs_vl_uni_models 包入口。

此包基于简化架构，支持纯 PyTorch 加载和测试，避免依赖 detectron22。
"""

from .config import get_default_config
from .models import DDETRSVLUni

__all__ = [
    'get_default_config',
    'DDETRSVLUni',
]
