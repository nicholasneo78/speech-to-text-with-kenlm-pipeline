#!/bin/bash

python3 ../../task_finetuning.py \
    --docker_image "nicholasneo78/stt_with_kenlm_pipeline:v0.1.1" \
    --project_name "wavlm_kenlm_pipeline" \
    --task_name "finetuning_resume" \
    --dataset_name "librispeech_v2_finetuning_resume_wavlm" \
    --output_url "s3://experiment-logging" \
    --dataset_project "datasets/librispeech" \
    --dataset_pkl_task_id "fdb1e1471ebb4b8dbf4f599080401819" \
    --dataset_pretrained_task_id "4790c892d0e5433594450d3a933fa2fe" \
    --train_pkl "pkl/librispeech_train.pkl" \
    --dev_pkl "pkl/librispeech_dev.pkl" \
    --test_pkl "pkl/librispeech_test.pkl" \
    --input_processor_path "processor/" \
    --input_checkpoint_path "ckpt/" \
    --input_pretrained_model_path "saved_model/" \
    --output_processor_path "root/processor/" \
    --output_checkpoint_path "root/ckpt/" \
    --output_saved_model_path "root/saved_model/" \
    --max_sample_length 450000 \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 0.005 \
    --warmup_steps 1000 \
    --architecture "wavlm" \
    --queue 'compute' \
