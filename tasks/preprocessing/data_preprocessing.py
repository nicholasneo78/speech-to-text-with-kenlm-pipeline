# imports
import os
from os.path import join
import numpy as np
import pandas as pd
import json
import librosa
from pathlib import Path
import yaml
from tqdm import tqdm
import pickle
from datasets import Dataset, DatasetDict
import re
from num2words import num2words
from nltk import flatten

# generate the pkl from scatch with all the data required to build the DatasetDict for the finetuning step 
class GeneratePickleFromScratch():
    
    def __init__(self, root_folder, pkl_filename, audio_format, additional_preprocessing='magister'):
        self.root_folder = root_folder
        self.pkl_filename = pkl_filename
        self.audio_format = audio_format
        self.additional_preprocessing = additional_preprocessing

    # create new directory and ignore already created ones
    def create_new_dir(self, directory):
        try:
            os.mkdir(directory)
        except OSError as error:
            pass # directory already exists!

    # helper function to build the lookup table for the id and annotations from all the text files and return the table
    def build_lookup_table(self):
        #initiate list to store the id and annotations lookup
        split_list_frame = []

        # get all the annotations into a dataframe
        for root, subdirs, files in os.walk(self.root_folder):
            for file in files:
                if file.endswith(".txt"):
                    # add on to the code here
                    df = pd.read_csv(os.path.join(root, file), header=None)
                    df.columns = ['name']

                    for i,j in enumerate(df.name):
                        split_list = j.split(" ",1)
                        split_list_frame.append(split_list)

        df_new = pd.DataFrame(split_list_frame, columns=['id', 'annotations']) # id and annotations are just dummy headers here
        return df_new 


    # to input the text and detect if any digits exists, if there is, will convert the numbers into its word representation
    def get_text_from_number(self, text):
        # split sentence to list of words
        text_list = text.split()
        new_text_list = []
        
        for txt in text_list:
            
            # check if word is STRICTLY alphanumeric, not either one of it
            if (txt.isalnum()) and (not txt.isalpha()) and (not txt.isnumeric()):
                sep_alpha_numeric_list = []
                
                # iterate at the letter/digits level
                for letter_idx, letter in enumerate(list(txt)):
                    
                    # append original letter
                    sep_alpha_numeric_list.append(txt[letter_idx])
                    # print(sep_alpha_numeric_list)
                    # print(len(list(txt)))
                    # print(letter_idx)
                    
                    # compare the current indexed letter/digits and the next index letter/digits
                    if letter_idx != len(list(txt))-1 and ((txt[letter_idx].isalpha() and txt[letter_idx+1].isnumeric()) or (txt[letter_idx].isnumeric() and txt[letter_idx+1].isalpha())):                    
                        sep_alpha_numeric_list.append('%')
                
                # join the list of characters to a word again but with '%' separator
                new_text_list.append(''.join(sep_alpha_numeric_list))
                
            # if word is not STRICTLY alphanumeric, just append the word
            else:
                new_text_list.append(txt)
        
        # remove the separator '%'
        preprocessed_text = ' '.join(flatten(new_text_list)).replace('%', ' ')
        
        # split the text into list of words again
        preprocessed_text_list = preprocessed_text.split()
        
        # print(preprocessed_text_list)
        
        # check and see if the individual strings are digits or not
        for idx, t in enumerate(preprocessed_text_list):
            try:
                # if less than 100, will pronounced in the usual way => e.g 55 == fifty-five
                if float(t) <= 100:
                    preprocessed_text_list[idx] = num2words(t)
                # else pronounced by its individual number => e.g 119 => one one nine
                else:
                    sep_num_list = []
                    for k in list(t):
                        sep_num_list.append(num2words(k))
                    preprocessed_text_list[idx] = sep_num_list
            except:
                continue
                
        text_list_flat = flatten(preprocessed_text_list)
        return ' '.join(text_list_flat).upper()


    # all the text preprocessing
    def preprocess_text(self, df, base_path):

        clean_text = df.loc[df['id'] == base_path, 'annotations'].to_numpy()[0]
        
        # additional preprocessing to replace the filler words with one symbol
        if self.additional_preprocessing == 'magister':
            clean_text = clean_text.replace('#', ' ').replace('<FIL>', '#').replace('<FILL>', '#')
        
        # add more here for other filler word or additional preprocessing needed for other data
        # elif ...

        else:
            clean_text = clean_text.replace('#', ' ')

        # keep only certain characters
        clean_text = re.sub(r'[^A-Za-z0-9#\' ]+', ' ', clean_text)
        
        # replace hyphen with space because hyphen cannot be heard
        clean_text = clean_text.replace('-', ' ')

        # convert all the digits to its text equivalent
        clean_text = self.get_text_from_number(clean_text)

        # convert multiple spaces into only one space
        clean_text = ' '.join(clean_text.split())

        return clean_text


    # generate the pickle file from scratch to prepare the final dataset for finetuning step
    def build_pickle_from_scratch(self):

        # list to append all the data in
        data_list = []
        
        # build the lookup table
        df_new = self.build_lookup_table()
        
        # retrieve the dataframe for the lookup table and create the pkl file
        for root, subdirs, files in tqdm(os.walk(self.root_folder)):
            for _, file in enumerate(files):
                if file.endswith(self.audio_format):
                    
                    # retrieve the base path for the particular audio file
                    base_path = os.path.basename(os.path.join(root, file)).split('.')[0]
                    
                    # get the array of values from the audio files and using 16000 sampling rate (16000 due to w2v2 requirement)
                    audio_array, _ = librosa.load(os.path.join(root, file), sr=16000)
                    
                    # NOTE THAT THERE ARE TWO LEVEL OF DICTIONARY: 
                        # the sub dictionary for the audio component 
                        # the main dictionary which comprises the file, audio and text component
                    
                    # text preprocessing
                    clean_text = self.preprocess_text(df_new, base_path)

                    # creating the final data dictionary that is to be saved to a pkl file
                    data = {
                        'file': os.path.join(root, file),
                        'audio': {
                            'array': audio_array, 
                            'path': os.path.join(root, file), 
                            'sampling_rate': 16000
                        },
                        'text': clean_text
                    }
                    
                    data_list.append(data)
                    
        # form the dataframe
        df_final = pd.DataFrame(data_list)

        # create pickle folder if it does not exist
        self.create_new_dir('./pkl/')
        
        # export the dataframe to pickle
        df_final.to_pickle(self.pkl_filename)
        
        return df_final  
        
    def __call__(self):
        return self.build_pickle_from_scratch()


# generate the pkl from manifest with all the data required to build the DatasetDict for the finetuning step 
class GeneratePickleFromManifest():
    def __init__(self, manifest_path, pkl_filename, additional_preprocessing='magister_v2'):
        self.manifest_path = manifest_path
        self.pkl_filename = pkl_filename
        self.additional_preprocessing = additional_preprocessing

    # create new directory and ignore already created ones
    def create_new_dir(self, directory):
        try:
            os.mkdir(directory)
        except OSError as error:
            pass # directory already exists!

    # to input the text and detect if any digits exists, if there is, will convert the numbers into its word representation
    def get_text_from_number(self, text):
        # split sentence to list of words
        text_list = text.split()
        new_text_list = []
        
        for txt in text_list:
            
            # check if word is STRICTLY alphanumeric, not either one of it
            if (txt.isalnum()) and (not txt.isalpha()) and (not txt.isnumeric()):
                sep_alpha_numeric_list = []
                
                # iterate at the letter/digits level
                for letter_idx, letter in enumerate(list(txt)):
                    
                    # append original letter
                    sep_alpha_numeric_list.append(txt[letter_idx])
                    
                    # compare the current indexed letter/digits and the next index letter/digits
                    if letter_idx != len(list(txt))-1 and ((txt[letter_idx].isalpha() and txt[letter_idx+1].isnumeric()) or (txt[letter_idx].isnumeric() and txt[letter_idx+1].isalpha())):                    
                        sep_alpha_numeric_list.append('%')
                
                # join the list of characters to a word again but with '%' separator
                new_text_list.append(''.join(sep_alpha_numeric_list))
                
            # if word is not STRICTLY alphanumeric, just append the word
            else:
                new_text_list.append(txt)
        
        # remove the separator '%'
        preprocessed_text = ' '.join(flatten(new_text_list)).replace('%', ' ')
        
        # split the text into list of words again
        preprocessed_text_list = preprocessed_text.split()
        
        # print(preprocessed_text_list)
        
        # check and see if the individual strings are digits or not
        for idx, t in enumerate(preprocessed_text_list):
            try:
                # if less than 100, will pronounced in the usual way => e.g 55 == fifty-five
                if float(t) <= 100:
                    preprocessed_text_list[idx] = num2words(t)
                # else pronounced by its individual number => e.g 119 => one one nine
                else:
                    sep_num_list = []
                    for k in list(t):
                        sep_num_list.append(num2words(k))
                    preprocessed_text_list[idx] = sep_num_list
            except:
                continue
                
        text_list_flat = flatten(preprocessed_text_list)
        return ' '.join(text_list_flat).upper()


    # all the text preprocessing
    def preprocess_text(self, text):

        # additional preprocessing to replace the filler words with one symbol
        if self.additional_preprocessing == 'magister_v2':
            clean_text = text.replace('#', ' ').replace('<unk>', '#')
        
        # add more here for other filler word or additional preprocessing needed for other data
        # elif ...

        else:
            clean_text = text.replace('#', ' ')
        
        # keep only certain characters
        clean_text = re.sub(r'[^A-Za-z0-9#\' ]+', ' ', clean_text)
        
        # replace hyphen with space because hyphen cannot be heard
        clean_text = clean_text.replace('-', ' ')

        # convert all the digits to its text equivalent
        clean_text = self.get_text_from_number(clean_text)

        # convert multiple spaces into only one space
        clean_text = ' '.join(clean_text.split())

        return clean_text


    # generate the pickle file from manifest data file to prepare the final dataset for finetuning step
    def build_pickle_from_manifest(self):

        # dict_list: to create a list to store the dictionaries from the manifest file
        # data_list: to store the data into this list to be exported into a pkl file
        dict_list, data_list = [], []

        # load manifest file into a list of dictionaries
        with open(self.manifest_path, 'rb') as f:
            for line in f:
                dict_list.append(json.loads(line))

        # iterate through the data_list and create the final pkl dataset file
        for entries in tqdm(dict_list):

            # get the array of values from the audio files and using 16000 sampling rate (16000 due to w2v2 requirement)
            # split the rightmost / and only take the parent directory of the manifest file
            audio_array, _ = librosa.load(f"{self.manifest_path.rsplit('/', 1)[0]}/{entries['audio_filepath']}", sr=16000)

            # text preprocessing
            clean_text = self.preprocess_text(entries['text'])

            # creating the final data dictionary that is to be saved to a pkl file
            data = {'file': f"{self.manifest_path.rsplit('/', 1)[0]}/{entries['audio_filepath']}",
                    'audio': {
                        'array': audio_array,
                        'path': f"{self.manifest_path.rsplit('/', 1)[0]}/{entries['audio_filepath']}",
                        'sampling_rate': 16000
                    },
                    'text': clean_text
            }

            data_list.append(data)

        # form the dataframe
        df_final = pd.DataFrame(data_list)

        # create pickle folder if it does not exist
        self.create_new_dir('./pkl/')

        # export the dataframe to pickle
        df_final.to_pickle(self.pkl_filename)

        return df_final
    

    def __call__(self):
        return self.build_pickle_from_manifest()


if __name__ == "__main__":

    # # magister dataset v1
    # magister_train_pkl = GeneratePickleFromScratch(root_folder='./datasets/magister_data_flac_16000/train/', 
    #                                                pkl_filename='./pkl/magister_data_flac_16000_train.pkl', 
    #                                                audio_format='.flac',
    #                                                additional_preprocessing='magister')

    # magister_dev_pkl = GeneratePickleFromScratch(root_folder='./datasets/magister_data_flac_16000/dev/', 
    #                                              pkl_filename='./pkl/magister_data_flac_16000_dev.pkl', 
    #                                              audio_format='.flac',
    #                                              additional_preprocessing='magister')

    # magister_test_pkl = GeneratePickleFromScratch(root_folder='./datasets/magister_data_flac_16000/test/', 
    #                                               pkl_filename='./pkl/magister_data_flac_16000_test.pkl', 
    #                                               audio_format='.flac',
    #                                               additional_preprocessing='magister')

    # df_train = magister_train_pkl()
    # df_dev = magister_dev_pkl()
    # df_test = magister_test_pkl()

    # # magister dataset v2
    # magister_v2_train_pkl = GeneratePickleFromManifest(manifest_path='./datasets/magister_data_v2_wav_16000/train_manifest.json', 
    #                                                    pkl_filename='./pkl/magister_data_v2_wav_16000_train.pkl', 
    #                                                    additional_preprocessing='magister_v2')

    # magister_v2_dev_pkl = GeneratePickleFromManifest(manifest_path='./datasets/magister_data_v2_wav_16000/dev_manifest.json', 
    #                                                    pkl_filename='./pkl/magister_data_v2_wav_16000_dev.pkl', 
    #                                                    additional_preprocessing='magister_v2')

    # magister_v2_test_pkl = GeneratePickleFromManifest(manifest_path='./datasets/magister_data_v2_wav_16000/test_manifest.json', 
    #                                                    pkl_filename='./pkl/magister_data_v2_wav_16000_test.pkl', 
    #                                                    additional_preprocessing='magister_v2')

    # df_train = magister_v2_train_pkl()
    # df_dev = magister_v2_dev_pkl()
    # df_test = magister_v2_test_pkl()

    # librispeech dataset
    librispeech_train_pkl = GeneratePickleFromManifest(manifest_path='./datasets/librispeech_v2/train/train_manifest.json', 
                                                       pkl_filename='./pkl/librispeech_train.pkl', 
                                                       additional_preprocessing='magister_v2')

    librispeech_dev_pkl = GeneratePickleFromManifest(manifest_path='./datasets/librispeech_v2/dev/dev_manifest.json', 
                                                       pkl_filename='./pkl/librispeech_dev.pkl', 
                                                       additional_preprocessing='magister_v2')

    librispeech_test_pkl = GeneratePickleFromManifest(manifest_path='./datasets/librispeech_v2/test/test_manifest.json', 
                                                       pkl_filename='./pkl/librispeech_test.pkl', 
                                                       additional_preprocessing='magister_v2')

    df_train = librispeech_train_pkl()
    df_dev = librispeech_dev_pkl()
    df_test = librispeech_test_pkl()