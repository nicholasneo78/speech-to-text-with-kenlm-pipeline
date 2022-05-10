import numpy as np
import pandas as pd
import nltk
import os
import pickle
from tqdm import tqdm
import subprocess
 
class GetTxtFromPkl:
    '''
        loads the train and the dev pkl files to get the annotations and produce a text file with their combined annotations 
    '''

    def __init__(self, df_train_filepath: str, df_dev_filepath: str, txt_filepath: str) -> None:
        '''
            df_train_filepath: file path of the train pickle file
            df_dev_filepath: file path of the dev pickle file
            txt_filepath: file path of the text file with all the possible words in the train and the dev dataset to build the language model
        '''
        
        self.df_train_filepath = df_train_filepath
        self.df_dev_filepath = df_dev_filepath
        self.txt_filepath = txt_filepath

    def create_new_dir(self, directory: str) -> None:
        '''
            create new directory and ignore already created ones
        '''
        try:
            os.mkdir(directory)
        except OSError as error:
            pass # directory already exists!
    
    def load_pkl(self) -> pd.DataFrame:
        '''
            loading the train and the dev pickle dataset files
        '''
        with open(self.df_train_filepath, 'rb') as f:
            df_train = pickle.load(f)

        with open(self.df_dev_filepath, 'rb') as f:
            df_dev = pickle.load(f)

        return df_train, df_dev

    def generate_text_file(self) -> None:
        '''
            combining the annotations from the train and test pkl files and produce a txt file
        '''

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

    def __call__(self):
        return self.generate_text_file()

class BuildLM:
    '''
        get the text file and produce a kenlm arpa file
    '''

    def __init__(self, df_train_filepath: str, df_dev_filepath: str, script_path: str, root_path: str, txt_filepath: str, n_grams: int, dataset_name: str) -> None:
        '''
            df_train_filepath: file path of the train pickle file
            df_dev_filepath: file path of the dev pickle file
            script path: path of the build.sh script file to access kenlm
            root_path: the root directory where the kenlm is being installed
            txt_filepath: the text file path containing all the words in the train and dev pickle files
            n_grams: number of grams used for building the kenlm language model
            dataset_name: name of the generated kenlm language model
        '''
        
        self.df_train_filepath = df_train_filepath
        self.df_dev_filepath = df_dev_filepath
        self.script_path = script_path
        self.root_path = root_path # bash script param
        self.txt_filepath = txt_filepath # bash script param
        self.n_grams = n_grams  # bash script param
        self.dataset_name = dataset_name # bash script param

    def create_new_dir(self, directory: str) -> None:
        '''
            create new directory and ignore already created ones
        '''
        try:
            os.mkdir(directory)
        except OSError as error:
            pass # directory already exists!

    def build_lm(self):
        '''
            process to build the kenlm language model arpa file
        '''
        
        # generate the text file from the train and test pickle dataset
        get_txt_file = GetTxtFromPkl(df_train_filepath=self.df_train_filepath, 
                                     df_dev_filepath=self.df_dev_filepath, 
                                     txt_filepath=self.txt_filepath)

        # generate the text file
        get_txt_file()

        # create root folder if it does not exist
        self.create_new_dir('root/')
        self.create_new_dir('root/lm/')

        # pass arguments into the bash script after text file is generated
        subprocess.run(['chmod', '+x', self.script_path])
        subprocess.run(["bash", self.script_path, "-k", self.root_path, "-n", self.n_grams, "-d", self.dataset_name, "-t", self.txt_filepath])

        # returns the file path of the generated kenlm language model arpa file
        return f"root/lm/{self.n_grams}_gram_{self.dataset_name}.arpa"

    def __call__(self):
        return self.build_lm()

if __name__ == "__main__":

    get_lm = BuildLM(df_train_filepath='./root/pkl/librispeech_train.pkl',
                     df_dev_filepath='./root/pkl/librispeech_dev.pkl', 
                     script_path="./build_lm.sh", 
                     root_path="/stt_with_kenlm_pipeline/kenlm", 
                     txt_filepath="lm/librispeech_annotations.txt", 
                     n_grams="5", 
                     dataset_name="librispeech")

    lm_path = get_lm()