export CUDA_VISIBLE_DEVICES=0,1
python opensora/train/train_causalvae.py \
    --exp_name "causalvae_r128_bs16" \
    --batch_size 1 \
    --precision bf16 \
    --max_steps 40000 \
    --save_steps 100 \
    --video_path datasets/UCF-101 \
    --video_num_frames 17 \
    --resolution 128 \
    --sample_rate 1 \
    --n_nodes 1 \
    --devices 2 \
    --num_workers 8 \
    # --output_dir results/causalvae \
    # --load_from_checkpoint ./results/pretrained_488/
