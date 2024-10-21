#!/bin/bash
python3 predict.py --video_file $1 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction --batch_size 12