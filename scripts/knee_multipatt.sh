#!/bin/sh

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=disabled
export WANDB_API_KEY=***
export WANDB_DIR=/tmp/wandb
export WANDB_CACHE_DIR=/tmp/wandb_cache

uv run main.py --mode val \
	--name knee_multipatt \
	--model no_vn \
	--num_cascades 6 \
	--body_part knee \
	--experiment release \
	--crop_shape 320,320 \
	--in_shape 320,320 \
	--val_patterns equispaced_fraction magic random gaussian_2d poisson_2d radial_2d \
	--val_accelerations 4 \
	--sample_rate 0.1 \
	--ckpt_path weights/knee_multipatt.ckpt
