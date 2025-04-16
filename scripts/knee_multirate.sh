#!/bin/sh

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export WANDB_MODE=disabled
export WANDB_API_KEY=834807c3ce19310795bf38319e568644792946b0
export WANDB_DIR=/tmp/wandb
export WANDB_CACHE_DIR=/tmp/wandb_cache

uv run main.py --mode val \
	--name knee_multirate \
	--model no_vn \
	--num_cascades 6 \
	--body_part knee \
	--experiment release \
	--crop_shape 320,320 \
	--in_shape 320,320 \
	--val_patterns equispaced_fraction \
	--val_accelerations 4 6 8 16 \
	--sample_rate 0.1 \
	--ckpt_path weights/knee_multirate.ckpt
