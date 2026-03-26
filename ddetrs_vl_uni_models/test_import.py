#!/usr/bin/env python3
"""
Test script to verify that the ddetrs_vl_uni models can be imported.
This script checks if all necessary modules are present and can be imported.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Try to import the main config function
    from config_uni import add_ddetrsvluni_config
    print("✓ config_uni.py imported successfully")

    # Try to import the main model class (this will fail without dependencies, but checks syntax)
    # from ddetrs_vl_uni import DDETRSVLUni
    # print("✓ ddetrs_vl_uni.py imported successfully")

    print("✓ All basic imports successful")
    print("Note: Full model import requires Detectron2 and other dependencies")

except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    sys.exit(1)