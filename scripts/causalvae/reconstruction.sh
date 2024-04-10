CUDA_VISIBLE_DEVICES=0 python examples/rec_imvi_vae.py \
    --model_path results/causalvae \
    --video_path assets/dino_demo.mp4 \
    --rec_path rec.mp4 \
    --device cuda \
    --sample_rate 1 \
    --num_frames 65 \
    --resolution 128 \
    --crop_size 128 \
    --ae CausalVAEModel_4x8x8 \
    --enable_tiling
