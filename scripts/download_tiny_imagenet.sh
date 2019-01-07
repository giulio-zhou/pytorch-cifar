#!/bin/bash
TRAIN_DIR=tiny-imagenet-200/train
LABELS=tiny-imagenet-200/words.txt
VAL_DIR=tiny-imagenet-200/val/images
VAL_LABELS=tiny-imagenet-200/val/val_annotations.txt
OUTDIR='tiny_imagenet_data'

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
python scripts/download_tiny_imagenet.py extract_train_imgs $TRAIN_DIR $LABELS $OUTDIR/train
python scripts/download_tiny_imagenet.py extract_val_imgs $VAL_DIR $TRAIN_DIR $LABELS $VAL_LABELS $OUTDIR/val
