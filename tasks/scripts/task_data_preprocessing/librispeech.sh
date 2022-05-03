#!/bin/bash

python3 ../../task_data_preprocessing.py \
    --docker_image "nicholasneo78/stt_with_kenlm_pipeline:v0.1.1" \
    --project_name "wav2vec2_kenlm_pipeline" \
    --task_name "data_preprocessing" \
    --dataset_name "librispeech_v2_data_preprocessing" \
    --output_url "s3://experiment-logging" \
    --dataset_project "datasets/librispeech" \
    --dataset_task_id "0d7c03aeb2b546b0813fb85dc20ace65" \
    --manifest_path_train "train/train_manifest.json" \
    --pkl_train "root/pkl/librispeech_train.pkl" \
    --manifest_path_dev "dev/dev_manifest.json" \
    --pkl_dev "root/pkl/librispeech_dev.pkl" \
    --manifest_path_test "test/test_manifest.json" \
    --pkl_test "root/pkl/librispeech_test.pkl" \
    --additional_preprocessing "general" \
    --queue "compute"
