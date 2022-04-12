# imports
from distutils.command.clean import clean
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

# generate the csv with all the data required to build the DatasetDict for the finetuning step 
class GeneratePickle():
    
    def __init__(self, root_folder, pkl_filename, audio_format):
        self.root_folder = root_folder
        self.pkl_filename = pkl_filename
        self.audio_format = audio_format
        
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

        # replace filler words with one symbol: 
        clean_text = df.loc[df['id'] == base_path, 'annotations'].to_numpy()[0].replace('#', ' ').replace('<FIL>', '#').replace('<FILL>', '#')
        
        # keep only certain characters
        clean_text = re.sub(r'[^A-Za-z0-9#\' ]+', ' ', clean_text)
        
        # replace hyphen with space because hyphen cannot be heard
        clean_text = clean_text.replace('-', ' ')

        # convert all the digits to its text equivalent
        clean_text = self.get_text_from_number(clean_text)

        # convert multiple spaces into only one space
        clean_text = ' '.join(clean_text.split())

        return clean_text

    # generate the csv to prepare the dataset for the finetuning step
    def build_pickle(self):
        # list to append all the data in
        data_list = []
        
        # build the lookup table
        df_new = self.build_lookup_table()
        
        # retrieve the dataframe for the lookup table and create the csv file
        for root, subdirs, files in tqdm(os.walk(self.root_folder)):
            for _, file in enumerate(files):
                if file.endswith(self.audio_format):
                    
                    # retrieve the base path for the particular audio file
                    base_path = os.path.basename(os.path.join(root, file)).split('.')[0]
                    
                    # get the array of values from the audio files and using 16000 sampling rate (16000 due to w2v2 requirment)
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
        
        # export the dataframe to pickle
        df_final.to_pickle(self.pkl_filename)
        
        return df_final  
        
    def __call__(self):
        return self.build_pickle()

if __name__ == "__main__":
    # get the pkl dataset
    generate_pkl_train = GeneratePickle(root_folder='./datasets/magister_data_flac_16000/train/', 
                                        pkl_filename='./pkl/magister_data_flac_16000_train.pkl', 
                                        audio_format='.flac')

    generate_pkl_dev = GeneratePickle(root_folder='./datasets/magister_data_flac_16000/dev/', 
                                    pkl_filename='./pkl/magister_data_flac_16000_dev.pkl', 
                                    audio_format='.flac')

    generate_pkl_test = GeneratePickle(root_folder='./datasets/magister_data_flac_16000/test/', 
                                pkl_filename='./pkl/magister_data_flac_16000_test.pkl', 
                                audio_format='.flac')

    df_train = generate_pkl_train()
    df_dev = generate_pkl_dev()
    df_test = generate_pkl_test()

    # for daniel
    # generate_pkl = GeneratePickle(root_folder='./data/', 
    #                             pkl_filename='data.pkl', 
    #                             audio_format='.wav')
    # df = generate_pkl()