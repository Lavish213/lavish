"""
Lavish Bootstrapper — global sys.path fix for imports anywhere
"""
import os, sys
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)