import numpy as np
import pandas as pd
import nltk
import os
import pickle
from tqdm import tqdm
import subprocess

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

        # create root folder if it does not exist
        self.create_new_dir('root/')
        self.create_new_dir('root/lm/')

        # remove the '#' (filler words) as it is not useful in building the language model and write the annotations into a .txt file
        with open(self.txt_filepath, 'w+') as f:
            for idx, text in enumerate(df_for_building_lm.text):
                f.write(f"{text.replace('# ', '').replace('#', '')}\n")

        # return self.txt_filepath

    def __call__(self):
        return self.generate_text_file()

# get the text file and produce a kenlm arpa file
class BuildLM():
    def __init__(self, df_train_filepath, df_dev_filepath, script_path, root_path, txt_filepath, n_grams, dataset_name):
        self.df_train_filepath = df_train_filepath
        self.df_dev_filepath = df_dev_filepath
        self.script_path = script_path
        self.root_path = root_path # bash script param
        self.txt_filepath = txt_filepath # bash script param
        self.n_grams = n_grams  # bash script param
        self.dataset_name = dataset_name # bash script param

    # create new directory and ignore already created ones
    def create_new_dir(self, directory):
        try:
            os.mkdir(directory)
        except OSError as error:
            pass # directory already exists!

    def build_lm(self):
        get_txt_file = GetTxtFromPkl(df_train_filepath=self.df_train_filepath, 
                                     df_dev_filepath=self.df_dev_filepath, 
                                     txt_filepath=self.txt_filepath)

        # generate the text file
        get_txt_file()

        print('\nGet text file completed\n')

        # create root folder if it does not exist
        self.create_new_dir('root/')
        self.create_new_dir('root/lm/')

        # pass arguments into the bash script after text file is generated
        #subprocess.run(['chmod', '+x', self.script_path])
        subprocess.run(["bash", self.script_path, "-k", self.root_path, "-n", self.n_grams, "-d", self.dataset_name, "-t", self.txt_filepath])
        
        print('\nExecution of script completed\n')

        return f"root/lm/{self.n_grams}_gram_{self.dataset_name}.arpa"
        #return self.output_arpa

    def __call__(self):
        return self.build_lm()

if __name__ == "__main__":

    get_lm = BuildLM(df_train_filepath='./root/pkl/magister_data_v2_wav_16000_train.pkl',
                     df_dev_filepath='./root/pkl/magister_data_v2_wav_16000_dev.pkl', 
                     script_path="./build_lm.sh", 
                     root_path="/stt_with_kenlm_pipeline", 
                     txt_filepath="lm/magister_v2_annotations.txt", 
                     n_grams="5", 
                     dataset_name="magister_v2")

    lm_path = get_lm()