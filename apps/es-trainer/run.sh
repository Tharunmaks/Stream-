#!/bin/sh
# Box:    ./run.sh            (PORT=8765 default)
# Termux: pkg install python clang && pip install flask; build es_lora/es_chat; ES_BIN_DIR=~/es ES_MODEL_DIR=~/models ./run.sh
cd "$(dirname "$0")"; exec python3 app.py
