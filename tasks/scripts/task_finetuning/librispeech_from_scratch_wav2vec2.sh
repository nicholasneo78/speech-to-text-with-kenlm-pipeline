#!/bin/bash

python3 ../../task_finetuning.py \
    --project_name "wav2vec2_kenlm_pipeline" \
    --task_name "finetuning" \
    --dataset_name "librispeech_v2_finetuning_wav2vec2" \
    --output_url "s3://experiment-logging" \
    --dataset_project "datasets/librispeech" \
    --dataset_pkl_task_id "fdb1e1471ebb4b8dbf4f599080401819" \
    --dataset_pretrained_task_id "c407832abe274bdfadbee1615e400bd2" \
    --train_pkl "pkl/librispeech_train.pkl" \
    --dev_pkl "pkl/librispeech_dev.pkl" \
    --test_pkl "pkl/librispeech_test.pkl" \
    --input_processor_path "root/processor/" \
    --input_checkpoint_path "root/ckpt/" \
    --input_pretrained_model_path "wav2vec2_base_model/" \
    --output_processor_path "root/processor/" \
    --output_checkpoint_path "root/ckpt/" \
    --output_saved_model_path "root/saved_model/" \
    --max_sample_length 450000 \
    --batch_size 16 \
    --epochs 10 \
    --lr 1e-4 \
    --weight_decay 0.005 \
    --warmup_steps 1000 \
    --architecture "wav2vec2" \
    --queue 'compute' \
    --finetune_from_scratch 
