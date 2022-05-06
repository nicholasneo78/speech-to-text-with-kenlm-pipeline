# Speech-to-Text with Kenlm Pipeline
A self-supervised speech-to-text pipeline that is finetuned on HuggingFace's wav2vec2 and wavlm pretrained models. The finetuned models produced from this pipeline is then evaluated with the use of `Kenlm` language model and beam search capability from the `pyctcdecode` libraries. This repository is centered around the ClearML MLOps Framework, but running the code on the local machine is also possible.  
  
## Introduction 
### Research Papers  
  
[wav2vec 2.0 Architecture](https://arxiv.org/abs/2006.11477) | [wavLM Architecture](https://arxiv.org/abs/2110.13900)   
   
### Tasks in this pipeline   
The tasks in this pipeline are as follows:  
1. Data Preprocessing  
2. Building the language model using Kenlm  
3. Finetuning the pretrained wav2vec2 and wavlm models from HuggingFace   
4. Evaluation of the finetuned model with the Kenlm language model built   
  
<br>  

## Project Organization  
There are some folders that are required to be created to store your speech and audio datasets and the base models used, the instructions will be shown below. You can get the base [wav2vec 2.0](https://huggingface.co/facebook/wav2vec2-base/tree/main) and [wavLM](https://huggingface.co/microsoft/wavlm-base/tree/main) models from the HuggingFace website.

The repository structure will be as shown below:  
```
    ├── README.md          <- The top-level README for developers using this project.
    ├── tasks
    |     
    │
    ├── dockerfile         <- The dockerfile to build an image with the required packages
    │
    └── requirements.txt   <- The requirements file for reproducing project environment
```

<br>

# Executing code on local machine
**The documentations below are for running the code using local machine, please go to the section on "Executing code on ClearML" if you want to run your code on ClearML**  

### Getting Started - Via Docker 
**Preferably, a Linux OS should be used**
1. Ensure that docker is installed in your computer
2. Clone this repository
```shell
git clone https://github.com/nicholasneo78/speech-to-text-with-kenlm-pipeline
```
3. Open the terminal and go to the root directory of this repository
4. Build the Dockerfile that is created in the repository using the docker-compose.yml file
```shell
docker-compose up --build -d
```
5. After building the docker image, check if the image is successfully built, by typing the following command
```shell
docker images
```
You should see the image `stt-with-kenlm-pipeline` with the tag `latest` in the list of docker images  
  
### Entering the docker image
1. To enter into the docker image to execute codes in the pipeline, execute this command
```shell
docker-compose run local bash
```
The codes are then ready to be executed inside the docker image, more information about executing each code will be discussed below.  
   
<br>   
   
## Data Preprocessing  
To preprocess the audio and the annotation data, split it into train-dev-test sets and convert into pickle files as required by the finetuning step. You can choose to generate the pickle datasets from scratch or from a manifest file. Check out [this repository](https://github.com/nicholasneo78/manifest-preprocessing) to see how you can generate the manifest file from your dataset.     
    
#### Arguments  

*Generate from scratch*  
- `root_folder`: (str) the folder path where all the audio files and annotations are found (".wav", ".flac", etc.)  
- `pkl_filename`: (str) the directory where the pickle file is being generated  
- `audio_format`: (str) the audio format of the targeted audio files, currently, only single audio extension per root folder is supported   
  
*Generate from manifest*  
- `manifest_path`: (str) the path where the manifest file resides (".json" format)  
- `pkl_filename`: (str) the directory where the pickle file is being generated    

#### Return
Both classes returns the pandas DataFrame and the pickle filepath where the pickle file is generated from the preprocessing.  
   
#### Before executing the code
Before executing the code, check the script `speech-to-text-with-kenlm-pipeline/tasks/preprocessing/data_preprocessing.py`, go to the bottom of the code, after the `if __name__ == "__main__"` line, call the class, either `GeneratePickleFromScratch` or `GeneratePickleFromManifest` to do the data preprocessing, here is a code snippet to illustrate the data preprocessing step:  
  
*Generate from scratch*  
```python
generate_train_pkl = GeneratePickleFromScratch(root_folder='<YOUR_TRAIN_DATASET_ROOT_FOLDER>', 
                                               pkl_filename='./root/pkl/<YOUR_FILENAME_OF_THE_TRAIN_PICKLE_FILE_GENERATED>.pkl', 
                                               audio_format='.<THE_AUDIO_FORMAT_OF_THE_AUDIO_FILES>')

generate_dev_pkl = GeneratePickleFromScratch(root_folder='<YOUR_DEV_DATASET_ROOT_FOLDER>', 
                                             pkl_filename='./root/pkl/<YOUR_FILENAME_OF_THE_DEV_PICKLE_FILE_GENERATED>.pkl', 
                                             audio_format='.<THE_AUDIO_FORMAT_OF_THE_AUDIO_FILES>')

generate_test_pkl = GeneratePickleFromScratch(root_folder='<YOUR_TEST_DATASET_ROOT_FOLDER>', 
                                              pkl_filename='./root/pkl/<YOUR_FILENAME_OF_THE_TEST_PICKLE_FILE_GENERATED>.pkl', 
                                              audio_format='.<THE_AUDIO_FORMAT_OF_THE_AUDIO_FILES>')

train_df, train_pkl_path = generate_train_pkl()
dev_df, dev_pkl_path = generate_dev_pkl()
test_df, test_pkl_path = generate_test_pkl()
```  
   
*Generate from manifest*  
```python
generate_train_pkl = GeneratePickleFromManifest(manifest_path='<YOUR_TRAIN_DATASET_MANIFEST_JSON_FILE>', 
                                                pkl_filename='./root/pkl/<YOUR_FILENAME_OF_THE_TRAIN_PICKLE_FILE_GENERATED>.pkl')

generate_dev_pkl = GeneratePickleFromManifest(manifest_path='<YOUR_DEV_DATASET_MANIFEST_JSON_FILE>',
                                              pkl_filename='./root/pkl/<YOUR_FILENAME_OF_THE_DEV_PICKLE_FILE_GENERATED>.pkl')

generate_test_pkl = GeneratePickleFromManifest(manifest_path='<YOUR_TEST_DATASET_MANIFEST_JSON_FILE>',
                                               pkl_filename='./root/pkl/<YOUR_FILENAME_OF_THE_TEST_PICKLE_FILE_GENERATED>.pkl')

train_df, train_pkl_path = generate_train_pkl()
dev_df, dev_pkl_path = generate_dev_pkl()
test_df, test_pkl_path = generate_test_pkl()
```  
  
There will be an example of the code tested on the librispeech dataset in the python script.    

#### Executing the code
To execute the data preprocessing code, on the terminal, go to this repository and enter into the docker image (refer above for the command), inside the docker container, type the following command:  
```shell
cd /stt-with-kenlm-pipeline/tasks/preprocessing
python3 data_preprocessing.py
```
<br>


## Building the language model using Kenlm  
To build a kenlm language model from the train and dev pickle files that were generated from the data preprocessing step. **Note: Do not pass in the test pickle file into building the language model as this will cause data leakage, causing inaccuracies in the evaluation phase.** The script will first load the train and dev pickle files, then will write all the annotations into a text file. It will then load the text file generated into the kenlm script to build the language model based on the train and dev dataset.  
  
#### Arguments  
- `df_train_filepath`: (str) the file path where the train pickle file is being generated    
- `df_dev_filepath`: (str) the file path where the dev pickle file is being generated   
- `script_path`: (str) script path to execute the building of the kenlm language model .arpa file  
- `root_path`: the root path where the kenlm directory and its packages resides  
- `txt_filepath`: (str) the text file path generated when the train and dev pickle file annotations are being consolidated    
- `n_grams`: (str) number of grams you want for the language model    
- `dataset_name`: (str) the name you want to give for the language model built    

#### Return
The filepath of the language model built.  
   
#### Before executing the code
Before executing the code, check the script `speech-to-text-with-kenlm-pipeline/tasks/preprocessing/build_lm.py`, go to the bottom of the code, after the `if __name__ == "__main__"` line, call the class `BuildLM` to build the Kenlm language model, here is a code snippet to illustrate the building of language model:   
   
```python
get_lm = BuildLM(df_train_filepath='./root/pkl/<YOUR_FILENAME_OF_THE_TRAIN_PICKLE_FILE_GENERATED>.pkl',
                 df_dev_filepath='./root/pkl/<YOUR_FILENAME_OF_THE_DEV_PICKLE_FILE_GENERATED>.pkl.pkl', 
                 script_path="./build_lm.sh", 
                 root_path="/stt_with_kenlm_pipeline/kenlm", 
                 txt_filepath="root/lm/<YOUR_TEXT_FILE_NAME>.txt", 
                 n_grams="<NUMBER_OF_GRAMS>", # preferably "5" 
                 dataset_name="<NAME OF THE DATASET>")

lm_path = get_lm()
```  
   
There will be an example of the code tested on the librispeech dataset in the python script.    

#### Executing the code
To execute the language model building code, on the terminal, go to this repository and enter into the docker image (refer above for the command), inside the docker container, type the following command:  
```shell
cd /stt-with-kenlm-pipeline/tasks/preprocessing
python3 build_lm.py
```
<br>

## Finetuning the pretrained wav2vec2 and wavlm models from HuggingFace 
The code to finetune the pretrained wav2vec2 and wavlm models from HuggingFace. In this repository, for demonstration purpose, only the base wav2vec2 and base wavlm pretrained models are used, but feel free to add the larger wav2vec2 and wavlm models as the base pretrained model. This script also has the ability to check for the audio length distribution of the train dataset, so that you can remove the longer audio data to prevent out-of-memory issues. *This script can also do evaluation on the test dataset but with the absence of a language model. Hence, it is not encouraged to get the evaluation score here, but rather on the evaluation script discussed later.*    

#### Arguments  
*Get audio length distribution*  
- `train_pkl`: (str) the file path where the train pickle file is being generated 
- `dev_pkl`: (str) the file path where the dev pickle file is being generated  
- `test_pkl`: (str) the file path where the test pickle file is being generated    
- `processor_path`: (str) the path where the processor will be generated from the loaded pickle file
- `max_sample_length`: (int) the maximum audio sample length of the audio file that could be accepted before hitting an out-of-memory issue (NOT USED because this is just checking the audio distribution and not the actual finetuning) 
- `mode`: (str) a flag to set, whether you want finetuning or checking audio distribution mode  
<br>  

*Finetuning*  
- `train_pkl`: (str) the file path where the train pickle file is being generated 
- `dev_pkl`: (str) the file path where the dev pickle file is being generated  
- `test_pkl`: (str) the file path where the test pickle file is being generated 
- `input_processor_path`: (str) the path where the processor is being input for the finetuning 
- `input_checkpoint_path`: (str) the path where the checkpoint is being input for the finetuning
- `input_pretrained_model_path`: (str) the path where the saved model is being input for the finetuning 
- `output_processor_path`: the path where the processor is being output after the finetuning   
- `output_checkpoint_path`: (str) the path where the checkpoint is being output after the finetuning
- `output_saved_model_path`: (str) the path where the saved model is being output after the finetuning
- `max_sample_length`: (int) the maximum audio sample length of the audio file that could be accepted before hitting an out-of-memory issue
- `batch_size`: (int) the batch size used for the finetuning
- `epochs`: (int) the number of epochs used for the finetuning
- `lr`: (float) learning rate used for the finetuning
- `weight_decay`: (float) weight decay used for the finetuning
- `warmup_steps`: (int) number of finetuning steps used with a lower learning rate for the model to adapt to finetuning
- `architecture`: (str) choose whether to finetune on wav2vec2 or wavlm architecture
- `finetune_from_scratch`: (boolean) whether to finetune from the pretrained base model or to resume finetuning from previous checkpoint steps
<br> 
  
*Evaluation*  
- `dev_pkl`: (str) the file path where the dev pickle file is being generated  
- `test_pkl`: (str) the file path where the test pickle file is being generated
- `processor_path`: (str) the path where the processor is being output from the finetuning step 
- `saved_model_path`: (str) the path where the saved model is being output after the finetuning   
- `architecture`: (str) choose whether to finetune on wav2vec2 or wavlm architecture
  
#### Return
*Get audio length distribution*   
None
  
*Finetuning*  
The output checkpoint path, output processor path, input pretrained model path and the output saved model path   
  
*Evaluation*   
None   


#### Before executing the code
Before executing the code, check the script `speech-to-text-with-kenlm-pipeline/tasks/preprocessing/build_lm.py`, go to the bottom of the code, after the `if __name__ == "__main__"` line, depending on your task, call the class `FinetuningPreparation` to get the audio length distribution, `Finetuning` to finetune the model and `Evaluation` to evaluate on the finetuned model. Here is a code snippet to illustrate the building of language model:  
<br>  

*Get audio length distribution*  
```python
distribution = FinetuningPreparation(train_pkl='./root/pkl/<YOUR_FILENAME_OF_THE_TRAIN_PICKLE_FILE_GENERATED>.pkl',
                                     dev_pkl='./root/pkl/<YOUR_FILENAME_OF_THE_DEV_PICKLE_FILE_GENERATED>.pkl', 
                                     test_pkl='./root/pkl/<YOUR_FILENAME_OF_THE_TEST_PICKLE_FILE_GENERATED>.pkl',
                                     processor_path='./root/<PATH_WHERE_THE_PROCESSOR_RESIDES>', 
                                     max_sample_length=None, 
                                     mode='get_audio_length_distribution')

distribution()
```
<br>

*Finetuning*  
```python
finetune_model = Finetuning(train_pkl='./root/pkl/<YOUR_FILENAME_OF_THE_TRAIN_PICKLE_FILE_GENERATED>.pkl',
                            dev_pkl='./root/pkl/<YOUR_FILENAME_OF_THE_DEV_PICKLE_FILE_GENERATED>.pkl', 
                            test_pkl='./root/pkl/<YOUR_FILENAME_OF_THE_TEST_PICKLE_FILE_GENERATED>.pkl', 
                            input_processor_path='./root/<YOUR_INPUT_PROCESSOR_PATH>', 
                            input_checkpoint_path='./root/<YOUR_INPUT_CHECKPOINT_PATH>', 
                            input_pretrained_model_path='./root/<YOUR_INPUT_PRETRAINED_MODEL_PATH>',
                            output_processor_path='./root/<YOUR_OUTPUT_PROCESSOR_PATH>', 
                            output_checkpoint_path='./root/<YOUR_OUTPUT_CHECKPOINT_PATH>', 
                            output_saved_model_path='./root/<YOUR_OUTPUT_SAVED_MODEL_PATH>', 
                            max_sample_length=<YOUR_MAX_AUDIO_SAMPLE_LENGTH_NUMBER>, 
                            batch_size=<BATCH_SIZE>, 
                            epochs=<NUMBER_OF_EPOCHS>,
                            lr=<LEARNING_RATE>, 
                            weight_decay=<WEIGHT_DECAY>, 
                            warmup_steps=<WARMUP_STEPS>, 
                            architecture='<EITHER_wav2vec2_OR_wavlm>',
                            finetune_from_scratch=<EITHER_True_OR_False>)

output_ckpt_path, output_processor_path, input_pretrained_model_path, output_saved_model_path = finetune_model()
```
<br>

*Evaluation*  
```python
evaluation = Evaluation(dev_pkl='./root/pkl/<YOUR_FILENAME_OF_THE_DEV_PICKLE_FILE_GENERATED>.pkl', 
                        test_pkl='./root/pkl/<YOUR_FILENAME_OF_THE_TEST_PICKLE_FILE_GENERATED>.pkl',
                        processor_path='./root/<YOUR_OUTPUT_PROCESSOR_PATH>', 
                        saved_model_path='./root/<YOUR_OUTPUT_SAVED_MODEL_PATH>',
                        architecture='<EITHER_wav2vec2_OR_wavlm>')

evaluation()
```

There will be an example of the code tested on the librispeech dataset in the python script.   
