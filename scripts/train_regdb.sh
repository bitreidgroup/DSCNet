#!/bin/bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python scripts/train.py --config configs/regdb.yaml