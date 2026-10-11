#!/usr/bin/env python3
"""Shortcut to Expert Craft. The real code lives in skills/expert-craft/scripts/ so that the same folder works as an installable skill."""
import os, runpy, sys

real = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills", "expert-craft", "scripts", "expertcraft.py")
sys.argv[0] = real
runpy.run_path(real, run_name="__main__")
