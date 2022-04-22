#!/bin/bash

# magister 
# n_gram="5"
# dataset_name="magister_v2"
# txtfile_loc="lm/magister_v2_annotations.txt"

# librispeech
# n_gram="5"
# dataset_name="librispeech"
# txtfile_loc="lm/librispeech.txt"
# kenlm_loc='/workspace'

while getopts n:d:t:k: flag
do
    case "${flag}" in
        n) n_gram=${OPTARG};;
        d) dataset_name=${OPTARG};;
        t) txtfile=${OPTARG};;
        k) kenlm_loc=${OPTARG};;
    esac
done

echo "Building the LM"

output_arpa_loc="root/lm/${n_gram}_gram_${dataset_name}.arpa"

# script execution
# "/workspace/kenlm/build/bin/lmplz" -o $n_gram < $txtfile > $output_arpa_loc
${kenlm_loc}/kenlm/build/bin/lmplz -o $n_gram < $txtfile > $output_arpa_loc