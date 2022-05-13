#!/bin/bash

python3 ../../task_data_sampling.py \
    --docker_image "nicholasneo78/stt_with_kenlm_pipeline:latest" \
    --project_name "wav2vec2_kenlm_pipeline" \
    --task_name "data_sampling" \
    --dataset_name "librispeech_v2_data_sampling" \
    --output_url "s3://experiment-logging" \
    --dataset_project "datasets/librispeech" \
    --dataset_task_id "fdb1e1471ebb4b8dbf4f599080401819" \
    --input_train_dict {"pkl/librispeech_train.pkl": 0.5, "pkl/librispeech_train.pkl": 0.2} \
    --input_dev_dict {"pkl/librispeech_dev.pkl": 1, "pkl/librispeech_dev.pkl": 1} \
    --input_test_dict {"pkl/librispeech_test.pkl": 1, "pkl/librispeech_test.pkl": 1} \
    --output_pkl_train "root/pkl/librispeech_train.pkl" \
    --output_pkl_dev "root/pkl/librispeech_dev.pkl" \
    --output_pkl_test "root/pkl/librispeech_test.pkl" \
    --sampling_mode "manual" \
    --random_state 42 \