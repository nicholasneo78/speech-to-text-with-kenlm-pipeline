#!/bin/bash

python3 ../../task_build_lm.py \
    --project_name "wav2vec2_kenlm_pipeline" \
    --task_name "build_lm" \
    --dataset_name "librispeech_v2_build_lm" \
    --output_url "s3://experiment-logging" \
    --dataset_project "datasets/librispeech" \
    --dataset_pkl_task_id "fdb1e1471ebb4b8dbf4f599080401819" \
    --script_task_id "67b056fb6b4a42b9b6704efa7f801f7d" \
    --kenlm_id "e429669809e5486784f0ace5cf182adb" \
    --train_pkl "pkl/librispeech_train.pkl" \
    --dev_pkl "pkl/librispeech_dev.pkl" \
    --script_path "build_lm.sh" \
    --txt_filepath "root/lm/librispeech.txt" \
    --n_grams "5" \
    --dataset_name_ "librispeech" \
    --queue "cpu-only"