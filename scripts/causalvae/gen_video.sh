python examples/rec_video_vae.py \
    --batch_size 1 \
    --real_video_dir test_eval/videos_test \
    --generated_video_dir test_eval/causalvae_r256_bs1_clip5_saveckpt_200k_videos_rec \
    --device cuda \
    --sample_fps 10 \
    --sample_rate 1 \
    --num_frames 17 \
    --resolution 256 \
    --num_workers 8 \
    --ckpt results/causalvae_r256_bs1_clip5_saveckpt_200k \
    --enable_tiling
    # --ckpt hf_models/Open-Sora-Plan-v1.0.0/vae \