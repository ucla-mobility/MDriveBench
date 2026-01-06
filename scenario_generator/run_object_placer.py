#!/usr/bin/env python3
"""
run_object_placer.py

Wrapper for pipeline/step_05_object_placer.
"""

from pipeline.step_05_object_placer.assets import *
from pipeline.step_05_object_placer.constants import *
from pipeline.step_05_object_placer.csp import *
from pipeline.step_05_object_placer.filters import *
from pipeline.step_05_object_placer.geometry import *
from pipeline.step_05_object_placer.guardrails import *
from pipeline.step_05_object_placer.main import main, run_object_placer
from pipeline.step_05_object_placer.model import *
from pipeline.step_05_object_placer.nodes import *
from pipeline.step_05_object_placer.parsing import *
from pipeline.step_05_object_placer.path_extension import *
from pipeline.step_05_object_placer.prompts import *
from pipeline.step_05_object_placer.spawn import *
from pipeline.step_05_object_placer.utils import *
from pipeline.step_05_object_placer.viz import *


if __name__ == "__main__":
    main()
