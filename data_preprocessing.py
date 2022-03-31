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

# generate the csv with all the data required to build the DatasetDict for the finetuning step 
class GeneratePickle():
    
    def __init__(self, root_folder, csv_filename, audio_format):
        self.root_folder = root_folder
        self.csv_filename = csv_filename
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
                    
                    # creating the dictionary
                    data = {
                        'file': os.path.join(root, file),
                        'audio': {
                            'array': audio_array, 
                            'path': os.path.join(root, file), 
                            'sampling_rate': 16000
                        },
                        'text': df_new.loc[df_new['id'] == base_path, 'annotations'].to_numpy()[0].replace('<FIL>', '&').replace('<FILL>', '&').replace('  ', ' ')
                    }
                    
                    data_list.append(data)
                    
        # form the dataframe
        df_final = pd.DataFrame(data_list)
        
        # export the dataframe to csv
        df_final.to_pickle(self.csv_filename)
        
        return df_final  
        
    def __call__(self):
        return self.build_pickle()

if __name__ == "__main__":
    # get the pkl dataset
    generate_pkl_train = GeneratePickle(root_folder='./datasets/magister_data_flac_16000_finetune/train/', 
                                        csv_filename='./pkl/magister_data_flac_16000_train.pkl', 
                                        audio_format='.flac')

    generate_pkl_dev = GeneratePickle(root_folder='./datasets/magister_data_flac_16000_finetune/dev/', 
                                    csv_filename='./pkl/magister_data_flac_16000_dev.pkl', 
                                    audio_format='.flac')

    df_train = generate_pkl_train()
    df_dev = generate_pkl_dev()