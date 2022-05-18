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
from typing import Tuple

class GeneratePickleFromScratch:
    '''
        generates the pkl from scatch with all the data required to build the DatasetDict for the finetuning step
    '''
    
    def __init__(self, root_folder: str, pkl_filename: str, audio_format: str, label: str, additional_preprocessing: str = 'general') -> None:
        '''
            root_folder: the root folder where the audio file with audio_format will be processed
            pkl_filename: the file path where the pickle data file will reside after preprocessing
            audio_format: the targeted audio format of the audio that is to be processed
            additional_preprocessing: depending on the annotations given, are there any other additional preprocessing needed to standardize the annotations
        '''
        self.root_folder = root_folder
        self.pkl_filename = pkl_filename
        self.audio_format = audio_format
        self.additional_preprocessing = additional_preprocessing
        self.label = label

    def create_new_dir(self, directory: str) -> None:
        '''
            creates new directory and ignore already created ones

            directory: the directory path that is being created
        '''
        try:
            os.mkdir(directory)
        except OSError as error:
            pass # directory already exists!

    def build_lookup_table(self) -> pd.DataFrame:
        '''
            helper function to build the lookup table for the id and annotations from all the text files and return the lookup table
        '''
        
        # create a list to store the id and annotations lookup
        split_list_frame = []

        # get all the annotations into a dataframe to build a lookup table
        for root, subdirs, files in os.walk(self.root_folder):
            for file in files:
                if file.endswith(".txt"):
                    # reads the annotation text files
                    df = pd.read_csv(os.path.join(root, file), header=None)
                    df.columns = ['name']

                    for i,j in enumerate(df.name):
                        split_list = j.split(" ",1)
                        split_list_frame.append(split_list)

        # id and annotations are just dummy headers here
        df_new = pd.DataFrame(split_list_frame, columns=['id', 'annotations']) 

        # returns the annotations in a pandas dataframe (lookup table)
        return df_new 

    def get_text_from_number(self, text: str) -> str:
        '''
            helper function to input the text and detect if any digits exists, if there is, will convert the numbers into its word representation
            
            text : obtains an entry of the annotations to do the text preprocessing
        '''
        # split sentence to a list of words
        text_list = text.split()

        # append the processed words to this list per iteration
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
                
        # make lists of lists into just a list of words
        text_list_flat = flatten(preprocessed_text_list)

        # returns the preprocessed text, where all the numbers in annotations are converted to text
        return ' '.join(text_list_flat).upper()

    def preprocess_text(self, df: pd.DataFrame, base_path: str) -> str:
        '''
            all the text preprocessing being done for the annotations

            df: lookup table containing the annotations
            base_path: get the audio file name w/o the extension
        '''

        # retrieve the annotations from the dataframe (lookup table) 
        clean_text = df.loc[df['id'] == base_path, 'annotations'].to_numpy()[0]
        
        # additional preprocessing to replace the filler words with one symbol
        if self.additional_preprocessing == 'general':
            clean_text = clean_text.replace('#', ' ').replace('<FIL>', '#').replace('<FILL>', '#')
        
        # add more here for other filler word or additional preprocessing needed for other data
        # elif ...

        # usual preprocessing of the annotations
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

        # returns the preprocessed annotations
        return clean_text

    def build_pickle_from_scratch(self) -> Tuple[pd.DataFrame, str]:
        '''
            generate the pickle file from scratch to prepare the final dataset for finetuning step
        '''

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
                    
                    # THERE ARE TWO LEVEL OF DICTIONARY: 
                        # the sub dictionary for the audio component 
                        # the main dictionary which comprises the file, audio and text component
                    
                    # text preprocessing
                    clean_text = self.preprocess_text(df_new, base_path)

                    # creating the final data dictionary in this format that is to be saved to a pkl file
                    data = {
                        'file': os.path.join(root, file),
                        'audio': {
                            'array': audio_array, 
                            'path': os.path.join(root, file), 
                            'sampling_rate': 16000
                        },
                        'text': clean_text,
                        'label': self.label
                    }
                    
                    data_list.append(data)
                    
        # form the dataframe
        df_final = pd.DataFrame(data_list)

        # create pickle folder if it does not exist
        self.create_new_dir('./root/')
        self.create_new_dir('./root/pkl/')
        
        # export the dataframe to pickle
        df_final.to_pickle(self.pkl_filename)

        # returns the final preprocessed dataframe and the filepath of the pickle file
        return df_final, self.pkl_filename
        
    def __call__(self):
        return self.build_pickle_from_scratch()

class GeneratePickleFromManifest:
    '''
        generate the pkl from manifest with all the data required to build the DatasetDict for the finetuning step 
    '''
    def __init__(self, manifest_path: str, pkl_filename: str, label: str, additional_preprocessing: str='general') -> None:
        '''
            manifest_path: the path to retrieve the manifest file with the information of the audio path and annotations
            pkl_filename: the file path where the pickle data file will reside after preprocessing
            additional_preprocessing: depending on the annotations given, are there any other additional preprocessing needed to standardize the annotations
        '''
        self.manifest_path = manifest_path
        self.pkl_filename = pkl_filename
        self.additional_preprocessing = additional_preprocessing
        self.label=label

    def create_new_dir(self, directory: str) -> None:
        '''
            create new directory and ignore already created ones

            directory: the directory path that is being created
        '''

        try:
            os.mkdir(directory)
        except OSError as error:
            pass # directory already exists!

    def get_text_from_number(self, text: str) -> str:
        '''
            to input the text and detect if any digits exists, if there is, will convert the numbers into its word representation

            text : obtains an entry of the annotations to do the text preprocessing
        '''

        # split sentence to list of words
        text_list = text.split()

        # append the preprocessed text in this list
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
                
        # make lists of lists into just a list of words
        text_list_flat = flatten(preprocessed_text_list)

        # returns the preprocessed text, where all the numbers in annotations are converted to text
        return ' '.join(text_list_flat).upper()

    def preprocess_text(self, text: str) -> str:
        '''
            all the text preprocessing being done for the annotations

            text: text annotations required to be preprocessed            
        '''

        # additional preprocessing to replace the filler words with one symbol
        if self.additional_preprocessing == 'general':
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

        # returns the preprocessed text
        return clean_text

    def build_pickle_from_manifest(self) -> Tuple[pd.DataFrame, str]:
        '''
            generate the pickle file from manifest data file to prepare the final dataset for finetuning step
        '''

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
                    'text': clean_text,
                    'label': self.label
            }

            data_list.append(data)

        # form the dataframe
        df_final = pd.DataFrame(data_list)

        # create pickle folder if it does not exist
        self.create_new_dir('./root/')
        self.create_new_dir('./root/pkl/')

        # export the dataframe to pickle
        df_final.to_pickle(self.pkl_filename)

        # returns the final preprocessed dataframe and the filepath of the pickle file
        return df_final, self.pkl_filename
    

    def __call__(self):
        return self.build_pickle_from_manifest()


if __name__ == "__main__":

    # librispeech dataset
    librispeech_train_pkl = GeneratePickleFromManifest(manifest_path='./datasets/librispeech_v2/train/train_manifest.json', 
                                                       pkl_filename='./root/pkl/librispeech_train.pkl',
                                                       label='librispeech')

    librispeech_dev_pkl = GeneratePickleFromManifest(manifest_path='./datasets/librispeech_v2/dev/dev_manifest.json', 
                                                       pkl_filename='./root/pkl/librispeech_dev.pkl',
                                                       label='librispeech')

    librispeech_test_pkl = GeneratePickleFromManifest(manifest_path='./datasets/librispeech_v2/test/test_manifest.json', 
                                                       pkl_filename='./root/pkl/librispeech_test.pkl',
                                                       label='librispeech')

    df_train, _ = librispeech_train_pkl()
    df_dev, _ = librispeech_dev_pkl()
    df_test, _ = librispeech_test_pkl()