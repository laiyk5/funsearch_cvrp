#!/usr/bin/env python3
"""
列出所有实验结果

用法:
    python scripts/list_results.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.output_manager import print_results_summary

if __name__ == "__main__":
    print_results_summary("outputs")
