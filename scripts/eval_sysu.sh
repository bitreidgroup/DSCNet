#!/bin/bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python scripts/test.py --config configs/sysu_all.yaml --ckpt   # please add the path to your ckpt files first
python scripts/test.py --config configs/sysu_indoor.yaml --ckpt   # please add the path to your ckpt files first