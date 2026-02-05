#!/bin/bash
# Wrapper script to run custom evaluation with debug output enabled
# Usage: ./run_custom_eval_debug.sh [same args as run_custom_eval.py]
#
# Available debug flags (set any combination):
#   DEBUG_PDM=1         - Enable PDM metrics calculation debug output
#   DEBUG_SCENARIOMGR=1 - Enable scenario manager debug output
#   DEBUG_FINISH=1      - Enable scenario completion/finish debug output
#
# Examples:
#   DEBUG_PDM=1 ./run_custom_eval_debug.sh ...
#   DEBUG_PDM=1 DEBUG_FINISH=1 ./run_custom_eval_debug.sh ...

# Enable all debug flags by default (you can comment out ones you don't need)
export DEBUG_PDM=1
export DEBUG_SCENARIOMGR=1
export DEBUG_FINISH=1

python tools/run_custom_eval.py "$@"
