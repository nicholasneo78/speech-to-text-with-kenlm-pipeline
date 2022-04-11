#!/bin/bash
n_gram=5
input_txt=lm_model_prep.txt
output_arpa=lm/magister_lm.arpa

# script execution
/workspace/kenlm/build/bin/lmplz -o $n_gram <$input_txt >$output_arpa


# script execution
#/workspace/kenlm/build/bin/lmplz -o 5 <lm_model_prep.txt >magister_lm.arpa