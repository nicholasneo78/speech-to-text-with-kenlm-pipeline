import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

class DataSampling:
    '''
        Samples the train data, given its proportion and combines multiple data if multiple datasets are used
    '''

    def __init__(self, data_dict, sampling_mode, final_pkl_location, random_state=42):
        self.data_dict = data_dict
        self.sampling_mode = sampling_mode
        self.final_pkl_location = final_pkl_location
        self.random_state = random_state
        
    # sampling the amount of data based on fixed proportion 
    def data_sampling_manual(self):
        # create the main dataframe to store the combined dataframe if there are more than 1 data
        df_main = pd.DataFrame()

        # load the data based on the data dict
        for key in self.data_dict:
            with open(key, 'rb') as f:
                df = pickle.load(f)
            
            # sample the data according to the fraction of data provided and the random state
            df_sample = df.sample(frac=self.data_dict[key], random_state=self.random_state)

            # append the iterated dataframe into the main dataframe
            df_main = df_main.append(df_sample)

        # shuffle the combined dataset for consistency
        df_main = df_main.sample(frac=1, random_state=self.random_state)

        # convert the combined dataset to pickle format
        df_main.to_pickle(self.final_pkl_location)

        return self.final_pkl_location

    # sampling amount of data based on the dataset with the least data entries
    def data_sampling_least(self):
        # create the main dataframe to store the combined dataframe if there are more than 1 data
        df_main = pd.DataFrame()

        # a list to store the number of entries a dataset has
        count_entries = []

        # load the data based on the data dict
        for key in self.data_dict:
            with open(key, 'rb') as f:
                df = pickle.load(f)
            
            # need to store the number of entries of each dataset to get the minimum and hence, can sample with that number
            count_entries.append(df.shape[0])

        # load the data again to do the sampling
        for key in self.data_dict:
            with open(key, 'rb') as f:
                df = pickle.load(f)

            # sample the data according to the dataset with the least entries
            df_sample = df.sample(n=min(count_entries), random_state=self.random_state)

            # append the iterated dataframe into the main dataframe
            df_main = df_main.append(df_sample)

        # shuffle the combined dataset for consistency
        df_main = df_main.sample(frac=1, random_state=self.random_state)
        
        # convert the combined dataset to pickle format
        df_main.to_pickle(self.final_pkl_location)
        
        return self.final_pkl_location

    # determine the sampling mode, whether to do it manually or to truncate the data to the dataset with the least entries
    def __call__(self):
        if self.sampling_mode == 'manual':
            return self.data_sampling_manual()
        elif self.sampling_mode == 'least':
            return self.data_sampling_least()

if __name__ == "__main__":
    
    # manual sampling
    df_train = DataSampling(data_dict={'./root/pkl/librispeech_train.pkl': 0.5, 
                                       './root/pkl/magister_data_v2_wav_16000_train.pkl' : 0.2}, 
                            sampling_mode='manual', 
                            final_pkl_location='./root/pkl/combined_train.pkl', 
                            random_state=42)

    _ = df_train()

    # # least sampling
    # df_train = DataSampling(data_dict={'./root/pkl/librispeech_train.pkl': None, 
    #                                    './root/pkl/magister_data_v2_wav_16000_train.pkl' : None}, 
    #                         sampling_mode='least', 
    #                         final_pkl_location='./root/pkl/combined_train.pkl', 
    #                         random_state=42)

    # _ = df_train()
