"""Fusion mini-app workflow: the IGNOS inner loop turned down, driven by LLM agents.

All physics output is synthetic (random-weight stand-in models). Importing this package
imports no torch and no agent framework.
"""
from pathlib import Path

__version__ = "0.1.0"

PARENT_DIR = Path(__file__).resolve().parent / "parent"   # five unmodified FusionFM/Demo files
