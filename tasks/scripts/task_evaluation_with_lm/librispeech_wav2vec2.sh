#!/bin/bash

python3 ../../task_evaluation_with_lm.py \
    --project_name "wav2vec2_kenlm_pipeline" \
    --task_name "evaluation_with_lm" \
    --output_url "s3://experiment-logging" \
    --dataset_pkl_task_id "fdb1e1471ebb4b8dbf4f599080401819" \
    --dataset_finetuned_task_id "7942389c4d234a06a916e91e49009b1c" \
    --lm_id "3366eaba06054260a40b02c6f9277dce" \
    --test_pkl "pkl/librispeech_test.pkl" \
    --finetuned_model_path "saved_model/" \
    --input_processor_path "processor/" \
    --lm_path "lm/5_gram_librispeech.arpa" \
    --alpha 0.6 \
    --beta 1.0 \
    --architecture "wav2vec2" \
    --queue "compute"
