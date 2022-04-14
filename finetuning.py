# imports
import numpy as np
import pandas as pd
from pathlib import Path
import json
import yaml
from tqdm import tqdm
import pickle
import librosa
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import random
import os

import torch
from datasets import Dataset, DatasetDict, load_metric
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, TrainingArguments, Trainer
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}\n')

# a class for all the finetuning preparations done
class FinetuningPreparation():

    def __init__(self, train_pkl, dev_pkl, test_pkl, processor_path='./processor/', max_sample_length=450000, mode='finetuning_prep'):
        self.train_pkl = train_pkl
        self.dev_pkl = dev_pkl
        self.test_pkl = test_pkl
        self.processor_path = processor_path
        self.max_sample_length = max_sample_length
        self.mode = mode

    # load the pickle data file
    def load_pickle_data(self):
        with open(self.train_pkl, 'rb') as f:
            df_train = pickle.load(f)

        with open(self.dev_pkl, 'rb') as f:
            df_dev = pickle.load(f)
            
        with open(self.test_pkl, 'rb') as f:
            df_test = pickle.load(f)
            
        # make it into a DatasetDict Object
        dataset = DatasetDict({
            "train": Dataset.from_pandas(df_train),
            "dev": Dataset.from_pandas(df_dev),
            "test": Dataset.from_pandas(df_test)
        })

        return dataset

    # extract all characters available in the train and dev datasets
    def extract_all_chars(self, batch):
        all_text = " ".join(batch["text"])
        vocab = list(set(all_text))
        return vocab

    # prepare the processor object
    def build_processor(self, dataset):

        # extract characters from train dataset
        vocabs_train = self.extract_all_chars(dataset['train'])

        # extract characters from dev dataset
        vocabs_dev = self.extract_all_chars(dataset['dev'])

        # create a union of all distinct letters in the training and the dev datasets
        vocab_list = list(set(vocabs_train) | set(vocabs_dev))

        # convert resulting list into an enumerated dictionary
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}

        # replace space with a more visible character |
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]

        # add the [UNK], [PAD], bos and eos token
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)
        vocab_dict["<s>"] = len(vocab_dict)
        vocab_dict["</s>"] = len(vocab_dict)

        # make the useless vocabs as [UNK] in the end
        try:
            del vocab_dict["#"]
        except KeyError:
            pass
            
        try:
            del vocab_dict["-"]
        except KeyError:
            pass

        # renumber the dictionary values to fill up the blanks
        count = 0
        for key, value in vocab_dict.items():
            vocab_dict[key] = count
            count += 1
            
        # vocabulary is completed, now save the vocabulary as a json file
        with open('vocab.json', 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)

        # use the json file to instantiate an object of the Wav2Vec2CTCTokenizer class
        tokenizer = Wav2Vec2CTCTokenizer('vocab.json', unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", bos_token='<s>', eos_token='</s>')

        # after the tokenizer object is created, the vocab.json file is not needed anymore, since the processor file will be created and the vocab.json will be there, hence can remove it
        os.remove('vocab.json')

        # PREPARING THE FEATURE EXTRACTOR
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

        # wrap the feature extractor and tokenizer as a single Wav2VevProcessor class object
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        # save the processor
        processor.save_pretrained(self.processor_path)

        return processor

    # preprocess the dataset to feed into the transformer
    def preprocess_dataset_for_transformer(self, batch, processor):

        # proceed with the preprocessing of data
        audio = batch["audio"]

        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
        return batch

    # get the audio sample distribution - separate branch to check the distribution of the audio length of the train datasets in terms of sampling size
    def get_audio_length_distribution(self, dataset, processor):

        # further preprocessing of the dataset
        dataset = dataset.map(lambda x: self.preprocess_dataset_for_transformer(x, processor), remove_columns=dataset.column_names["train"], num_proc=1)

        # make a list to get the list of audio length of all the training data
        audio_length_list = []
        for idx, item in tqdm(enumerate(dataset['train'])):
            audio_length_list.append(dataset['train'][idx]['input_length'])

        # get the distribution of the audio sample
        data_dist = pd.Series(audio_length_list)

        data_dist.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
        plt.title('Distribution')
        plt.xlabel('Samples')
        plt.ylabel('Number of inputs')
        plt.grid(axis='y', alpha=0.75)
        plt.show()

    # filter the audio length to prevent OOM issue due to lengthy audio files
    def filter_audio_length(self, dataset):

        # filter out those longer duration videos (based on the histogram with the right tail minority)
        dataset["train"] = dataset["train"].filter(lambda x: x < self.max_sample_length, input_columns=["input_length"])

        return dataset

    # consolidating all the above methods for preparation
    def finetuning_preparation(self):
        
        # load the DatasetDict object from the pkl files 
        dataset = self.load_pickle_data()

        # prepare the processor object
        processor = self.build_processor(dataset)

        # preprocess the dataset to feed into the transformer
        dataset = dataset.map(lambda x: self.preprocess_dataset_for_transformer(x, processor), remove_columns=dataset.column_names["train"], num_proc=1)

        # filter the audio length to prevent OOM issue due to lengthy audio files
        dataset = self.filter_audio_length(dataset)

        return dataset

    # wrapper class of get_audio_length_distribution
    def get_audio_length_distribution_preparation(self):

        # load the DatasetDict object from the pkl files 
        dataset = self.load_pickle_data()

        # prepare the processor object
        processor = self.build_processor(dataset)

        # get the distribution of the train dataset
        self.get_audio_length_distribution(dataset, processor)


    def __call__(self):

        if self.mode == 'finetuning_prep':
            return self.finetuning_preparation()

        elif self.mode == 'get_audio_length_distribution':
            return self.get_audio_length_distribution_preparation()

### resume code here





if __name__ == "__main__":
    
    # get audio length distribution

    print('Getting the distribution of the train dataset\n')
    distribution = FinetuningPreparation(train_pkl='./pkl/magister_data_v2_wav_16000_train.pkl',
                                         dev_pkl='./pkl/magister_data_v2_wav_16000_dev.pkl', 
                                         test_pkl='./pkl/magister_data_v2_wav_16000_test.pkl',
                                         processor_path='./processor/', 
                                         max_sample_length=450000, 
                                         mode='get_audio_length_distribution')

    distribution()

    # get the prepared dataset for the finetuning task

    print('Getting the prepared dataset for the finetuning task\n')
    data_preparation = FinetuningPreparation(train_pkl='./pkl/magister_data_v2_wav_16000_train.pkl',
                                         dev_pkl='./pkl/magister_data_v2_wav_16000_dev.pkl', 
                                         test_pkl='./pkl/magister_data_v2_wav_16000_test.pkl',
                                         processor_path='./processor/', 
                                         max_sample_length=450000, 
                                         mode='finetuning_prep')

    dataset = data_preparation()


