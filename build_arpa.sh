#!/bin/bash
n_gram=5
txtfile_loc=lm/lm_model_prep.txt
output_arpa_loc=lm/magister_lm.arpa

# script execution
/workspace/kenlm/build/bin/lmplz -o $n_gram <$txtfile_loc >$output_arpa_loc


# script execution
#/workspace/kenlm/build/bin/lmplz -o 5 <lm_model_prep.txt >magister_lm.arpa