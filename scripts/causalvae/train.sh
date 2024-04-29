export CUDA_VISIBLE_DEVICES=2,3
python opensora/train/train_causalvae.py \
    --exp_name "causalvae_r256_bs1_clip5_saveckpt_200k" \
    --batch_size 1 \
    --precision bf16 \
    --max_steps 200000 \
    --save_steps 1000 \
    --video_path datasets/UCF-101 \
    --video_num_frames 5 \
    --resolution 256 \
    --sample_rate 1 \
    --n_nodes 1 \
    --devices 2 \
    --num_workers 8 \
    # --output_dir results/causalvae \
    # --load_from_checkpoint ./results/pretrained_488/
