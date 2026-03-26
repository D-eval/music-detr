# DDETRS VL Uni Models

This folder contains all the components for the DDETRSVLUni model from the Open-Det2 project.

## Structure

- `ddetrs_vl_uni.py`: Main model implementation (DDETRSVLUni class)
- `config_uni.py`: Configuration settings for DDETRSVLUni
- `__init__.py`: Package initialization
- `backbone/`: Backbone network implementations
- `models/`: Model components including deformable_detr, qfree_det, segmentation, and text processing
- `util/`: Utility functions for box operations, miscellaneous helpers
- `data/`: Data processing and dataset mappers

## Dependencies

This model depends on:
- Detectron2 (detectron22)
- PyTorch
- CLIP
- Other dependencies as listed in the original project's requirements

## Usage

To use this model, you need to integrate it with the Detectron2 framework. The main class is `DDETRSVLUni` which is registered in the META_ARCH_REGISTRY.

## Note

This is a copy of the original files from the Open-Det2 project for organizational purposes. The original project structure remains unchanged.