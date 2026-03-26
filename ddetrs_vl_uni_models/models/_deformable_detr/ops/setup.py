# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

import os
import glob
import sys

import torch

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]


def print_build_env():
    print("==== MultiScaleDeformableAttention build environment ====")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch CUDA version: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"CUDA_HOME: {CUDA_HOME}")
    print("========================================================")

def get_extensions():
    print_build_env()
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        print("Selected build type: CUDAExtension (WITH_CUDA)")
        print("Building MultiScaleDeformableAttention with CUDA support")
    else:
        # CPU-only build
        define_macros += [("WITHOUT_CUDA", None)]
        print("Selected build type: CppExtension (WITHOUT_CUDA)")
        if not torch.cuda.is_available():
            print("Reason: torch.cuda.is_available() is False")
        if CUDA_HOME is None:
            print("Reason: CUDA_HOME is None")
        print("Building MultiScaleDeformableAttention for CPU only (no CUDA support)")

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="MultiScaleDeformableAttention",
    version="1.0",
    author="Weijie Su",
    url="https://github.com/fundamentalvision/Deformable-DETR",
    description="PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)


# import sys
# sys.path.insert(0, '/Users/broyou/Desktop/Open-Det2/my/ddetrs_vl_uni_models/models/_deformable_detr/ops/build/lib.macosx-10.9-universal2-3.9')
# import MultiScaleDeformableAttention

# import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
# /Users/broyou/Library/Python/3.9/lib/python/site-packages/torch/lib
# export DYLD_LIBRARY_PATH=/Users/broyou/Library/Python/3.9/lib/python/site-packages/torch/lib:$DYLD_LIBRARY_PATH
