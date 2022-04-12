#!/bin/bash
n_gram="5"
dataset_name="magister"
txtfile_loc="lm/magister_text_build_lm.txt"

output_arpa_loc="lm/${n_gram}gram_${dataset_name}.arpa"

# script execution
/workspace/kenlm/build/bin/lmplz -o $n_gram < $txtfile_loc > $output_arpa_loc

# script execution
#/workspace/kenlm/build/bin/lmplz -o 5 <lm_model_prep.txt >magister_lm.arpa