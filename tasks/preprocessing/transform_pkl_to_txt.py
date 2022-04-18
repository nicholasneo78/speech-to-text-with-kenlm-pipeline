import numpy as np
import pandas as pd
import nltk
import os
import pickle
from tqdm import tqdm

# loads the train and the dev pkl files to get the annotations and produce a text file with their combined annotations  
class GetTxtFromPkl():
    def __init__(self, df_train_filepath, df_dev_filepath, txt_filepath):
        self.df_train_filepath = df_train_filepath
        self.df_dev_filepath = df_dev_filepath
        self.txt_filepath = txt_filepath

    # create new directory and ignore already created ones
    def create_new_dir(self, directory):
        try:
            os.mkdir(directory)
        except OSError as error:
            pass # directory already exists!
    
    def load_pkl(self):
        with open(self.df_train_filepath, 'rb') as f:
            df_train = pickle.load(f)

        with open(self.df_dev_filepath, 'rb') as f:
            df_dev = pickle.load(f)

        return df_train, df_dev

    # combining the annotations from the train and test pkl files and produce a txt file
    def generate_text_file(self):

        # load the train and dev pkl files
        df_train, df_dev = self.load_pkl()

        # combine the train and dev set to prep the creation of the language model
        df_for_building_lm = pd.concat([pd.DataFrame(df_train.text), pd.DataFrame(df_dev.text)]).reset_index(drop=True)

        # create pickle folder if it does not exist
        self.create_new_dir('./lm/')

        # remove the '#' (filler words) as it is not useful in building the language model and write the annotations into a .txt file
        with open(self.txt_filepath, 'w+') as f:
            for idx, text in enumerate(df_for_building_lm.text):
                f.write(f"{text.replace('# ', '').replace('#', '')}\n")

    def __call__(self):
        return self.generate_text_file()

if __name__ == "__main__":

    ########## MAGISTER CONFIG ##########

    # get_txt_from_pkl = GetTxtFromPkl(df_train_filepath='./pkl/magister_data_v2_wav_16000_train.pkl',
    #                                  df_dev_filepath='./pkl/magister_data_v2_wav_16000_dev.pkl',
    #                                  txt_filepath='lm/magister_v2_annotations.txt')

    # get_txt_from_pkl()

    ####################################################

    ########## LIBRISPEECH CONFIG ##########

    get_txt_from_pkl = GetTxtFromPkl(df_train_filepath='./pkl/librispeech_train.pkl',
                                     df_dev_filepath='./pkl/librispeech_dev.pkl',
                                     txt_filepath='lm/librispeech.txt')

    get_txt_from_pkl()

    ####################################################