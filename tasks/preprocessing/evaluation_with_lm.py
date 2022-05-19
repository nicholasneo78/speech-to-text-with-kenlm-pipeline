import kenlm
import pandas as pd
import numpy as np
from pyctcdecode import build_ctcdecoder
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, WavLMForCTC
import torch
import os
import re
import pickle
from jiwer import compute_measures
import datasets
from datasets import Dataset
from tqdm import tqdm
from copy import deepcopy

device = 'cuda' if torch.cuda.is_available else 'cpu'
print(f'Device: {device}\n')

class WER(datasets.Metric):
    '''
        WER metrics
    '''

    def __init__(self, predictions=None, references=None, concatenate_texts=False):
        self.predictions = predictions
        self.references = references
        self.concatenate_texts = concatenate_texts

    def compute(self):
        if self.concatenate_texts:
            return compute_measures(self.references, self.predictions)["wer"]
        else:
            incorrect = 0
            total = 0
            for prediction, reference in zip(self.predictions, self.references):
                measures = compute_measures(reference, prediction)
                incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
                total += measures["substitutions"] + measures["deletions"] + measures["hits"]
            return incorrect / total

class EvaluationWithLM:
    '''
        perform evaluation of the finetuned model with the test set that incorporates language model
    '''

    def __init__(self, finetuned_model_path: str, processor_path: str, lm_path: str, test_data_path: str, alpha: float, beta: float, architecture: str):
        '''
            finetuned_model_path: path to get the finetuned model
            processor_path: path to get the processor
            lm_path: path to get the language model arpa file
            test_data_path: path to get the test pickle data file for evaluation
            alpha: for decoding using language model - weight associated with the LMs probabilities. A weight of 0 means the LM has no effect
            beta: for decoding using language model - weight associated with the number of words within the beam
            architecture: either pretrained from wav2vec2 model or the wavlm model
        '''

        self.finetuned_model_path = finetuned_model_path
        self.processor_path = processor_path
        self.lm_path = lm_path
        self.test_data_path = test_data_path
        self.alpha = alpha
        self.beta = beta
        self.architecture = architecture

    def greedy_decode(self, logits: torch.Tensor, labels: str) -> str:
        '''
            greedy decode with no language model
        '''
        label_dict = {n: c for n, c in enumerate(labels)}
        prev_c = None
        out = []
        for n in logits.argmax(axis=1):
            c = label_dict.get(n, "")  
            # if not in labels, then assume it's ctc blank char
            if c != prev_c: 
                out.append(c)
            prev_c = c
        return "".join(out)

    def get_wer(self) -> float:
        '''
            get the final WER for both the greedy decoding and beam search decoding
        '''
        
        # load the finetuned model and the processor that is produced from finetuning.py script
        if self.architecture == 'wav2vec2':
            asr_model = Wav2Vec2ForCTC.from_pretrained(self.finetuned_model_path)
        elif self.architecture == 'wavlm':
            asr_model = WavLMForCTC.from_pretrained(self.finetuned_model_path)
        asr_processor = Wav2Vec2Processor.from_pretrained(self.processor_path)

        # get the vocab list from the dictionary
        vocab = list(asr_processor.tokenizer.get_vocab().keys())

        # convert some vocab tokens to see the text clearer when decoding
        vocab[vocab.index('[PAD]')] = '_'
        vocab[vocab.index('|')] = ' '

        # build the decoder and load the kenlm langauge model
        decoder = build_ctcdecoder(
                    labels = vocab,
                    kenlm_model_path = self.lm_path,
                    alpha = self.alpha, # weight associated with the LMs probabilities. A weight of 0 means the LM has no effect
                    beta = self.beta,  # weight associated with the number of words within the beam
                )

        # load the test dataset for evaluation
        with open(self.test_data_path, 'rb') as f:
            df_test = pickle.load(f)

        # convert the data into a huggingface Dataset object
        data_test = Dataset.from_pandas(df_test)

        # create dictionaries to separate the text into different dataset labels
        main_dict = {}

        # initialise the label (key) with an empty list (values)
        for label in list(np.unique(np.array(data_test['label']))):
            main_dict[label] = []
        
        # add another extra key value pair to get all the values to compute the overall WER
        main_dict['all'] = []

        # create copies for the different text annotatation
        ground_truth_dict = deepcopy(main_dict)
        pred_beam_dict = deepcopy(main_dict)
        pred_greedy_dict = deepcopy(main_dict)

        # append the text and predictions into lists
        for idx, entry in tqdm(enumerate(data_test)):
            # get logits
            audio_array = np.array(data_test[idx]['audio']['array'])
            input_values = asr_processor(audio_array, return_tensors="pt", sampling_rate=16000).input_values  
            logits = asr_model(input_values).logits.cpu().detach().numpy()[0]

            # ground truth
            ground_truth_text = data_test[idx]['text']

            # beam search decoding - can add hotwords and its weights too if needed
            beam_text = decoder.decode(logits)

            # greedy search decoding
            greedy_text = self.greedy_decode(logits, vocab)
            greedy_text = ("".join(c for c in greedy_text if c not in ["_"]))

            # store the ground truth and predicted annotations in dictionaries 
            ground_truth_dict[data_test['label'][idx]].append(ground_truth_text)
            pred_beam_dict[data_test['label'][idx]].append(beam_text)
            pred_greedy_dict[data_test['label'][idx]].append(greedy_text)

            # append to main key where data entries of all labels are combined
            ground_truth_dict['all'].append(ground_truth_text)
            pred_beam_dict['all'].append(beam_text)
            pred_greedy_dict['all'].append(greedy_text)

        # regex to obtain the number of grams for the final print message
        try:
            n_grams = re.search(r'\d+', os.path.basename(self.lm_path)).group()
            obtain_n = True
        except AttributeError:
            obtain_n = False

        # store the WER values into lists
        wer_greedy_dict = {}
        wer_beam_dict = {}

        # define the evaluation metric for the individual datasets and also the combined values
        for label in list(main_dict.keys()):
            
            # define evaluation metric
            wer_greedy = WER(predictions=pred_greedy_dict[label], references=ground_truth_dict[label])
            wer_beam = WER(predictions=pred_beam_dict[label], references=ground_truth_dict[label])

            # print the final evaluation
            if obtain_n:
                print(f'\n{n_grams}-gram | alpha = {self.alpha} | beta = {self.beta} | label: {label} | WER (greedy search): {wer_greedy.compute():.5f}')
                print(f'{n_grams}-gram | alpha = {self.alpha} | beta = {self.beta} | label: {label} | WER (beam search): {wer_beam.compute():.5f}\n')

            else:
                print(f'\nalpha = {self.alpha} | beta = {self.beta} | label: {label} | WER (greedy search): {wer_greedy.compute():.5f}')
                print(f'alpha = {self.alpha} | beta = {self.beta} | label: {label} | WER (beam search): {wer_beam.compute():.5f}\n')

            # append the wer values into the dictionaries
            wer_greedy_dict[label] = round(wer_greedy.compute(), 5)
            wer_beam_dict[label] = round(wer_beam.compute(), 5)


        # return the wer values for debugging
        return wer_greedy_dict, wer_beam_dict

    def __call__(self):
        return self.get_wer()

if __name__ == "__main__":

    # magister v2 wavlm
    evaluation = EvaluationWithLM(finetuned_model_path='./root/combined/wav2vec2/saved_model/', # or './root/librispeech/wav2vec2/saved_model/'
                                  processor_path='./root/combined/wav2vec2/processor/', # or './root/librispeech/wav2vec2/processor/'
                                  lm_path='root/lm/5_gram_combined.arpa', 
                                  test_data_path='./root/pkl/combined_test.pkl', 
                                  alpha=0.6, beta=1.0,
                                  architecture='wav2vec2')

    greedy, beam = evaluation()