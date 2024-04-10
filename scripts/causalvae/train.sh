python opensora/train/train_causalvae.py \
    --exp_name "causalvae_r32_bs1" \
    --batch_size 1 \
    --precision bf16 \
    --max_steps 40000 \
    --save_steps 100 \
    --video_path datasets/UCF-101 \
    --video_num_frames 1 \
    --resolution 32 \
    --sample_rate 1 \
    --n_nodes 1 \
    --devices 4 \
    --num_workers 8 \
    # --output_dir results/causalvae \
    # --load_from_checkpoint ./results/pretrained_488/
