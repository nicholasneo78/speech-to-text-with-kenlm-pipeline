#!/bin/bash
n_gram="5"
dataset_name="magister_v2"
txtfile_loc="lm/magister_v2_annotations.txt"

output_arpa_loc="lm/${n_gram}_gram_${dataset_name}.arpa"

# script execution
/workspace/kenlm/build/bin/lmplz -o $n_gram < $txtfile_loc > $output_arpa_loc