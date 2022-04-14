# imports
import numpy as np
import pandas as pd
from pathlib import Path
import json
import yaml
from tqdm import tqdm
import pickle
import librosa
import plotext as plt
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
        #data_dist = pd.Series(audio_length_list)
        #data_dist.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
        
        # change to plotext implementation instead of matplotlib
        plt.hist(audio_length_list, bins=50, label='train data')

        plt.title('Distribution')
        plt.xlabel('Samples')
        plt.ylabel('Number of inputs')
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

        return dataset, processor

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


# build a data collator class that uses ctc with padding
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


# set up trainer class to proceed with finetuning
class Finetuning():
    def __init__(self, train_pkl, dev_pkl, test_pkl, processor_path, checkpoint_path, pretrained_model_path, saved_model_path, max_sample_length, batch_size, epochs, lr, weight_decay, warmup_steps, finetune_from_scratch=False):
        self.train_pkl = train_pkl
        self.dev_pkl = dev_pkl
        self.test_pkl = test_pkl
        self.processor_path = processor_path
        self.checkpoint_path = checkpoint_path
        self.pretrained_model_path = pretrained_model_path
        self.saved_model_path = saved_model_path
        self.max_sample_length = max_sample_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.finetune_from_scratch = finetune_from_scratch
    
    # defining evaluation metric during finetuning process
    def compute_metrics(self, pred, processor):

        # load evaluation metric
        wer_metric = load_metric("wer")

        # get the predicted logits
        pred_logits = pred.predictions

        # get the predicted ids (character)
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        # decode
        pred_str = processor.batch_decode(pred_ids)

        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        # obtain metric score
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # proceed with finetuning of the model
    def finetune(self):

        # load the preprocessed dataset from the FinetuningPreparation class
        data_preparation = FinetuningPreparation(train_pkl=self.train_pkl,
                                                 dev_pkl=self.dev_pkl, 
                                                 test_pkl=self.test_pkl,
                                                 processor_path=self.processor_path, 
                                                 max_sample_length=self.max_sample_length, 
                                                 mode='finetuning_prep')
        
        if self.finetune_from_scratch:
            # obtain the preprocessed dataset and the processor
            dataset, processor = data_preparation()

            # load the pretrained model, and finetune from scratch (using wav2vec2_base_model from huggingface)
            model = Wav2Vec2ForCTC.from_pretrained(
                self.pretrained_model_path,
                ctc_loss_reduction="mean", 
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        else:
            # obtain only the preprocessed dataset and not the processor as it has already been built before, hence just load it
            dataset, _ = data_preparation()

            # to resume finetuning from checkpoints
            model = Wav2Vec2ForCTC.from_pretrained(self.pretrained_model_path)
            processor = Wav2Vec2Processor.from_pretrained(self.processor_path)

        # load the data collator that uses CTC with padding
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

        # setup training arguments
        training_args = TrainingArguments(
            output_dir=self.checkpoint_path,
            group_by_length=True,
            per_device_train_batch_size=self.batch_size,
            evaluation_strategy="steps",
            num_train_epochs=self.epochs,
            fp16=True,
            gradient_checkpointing=True,
            save_steps=200,
            eval_steps=200,
            logging_steps=200,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            save_total_limit=2,
            push_to_hub=False,
        )

        # defining the trainer class
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=lambda x : self.compute_metrics(x, processor),
            train_dataset=dataset["train"],
            eval_dataset=dataset["dev"],
            tokenizer=processor.feature_extractor,
        )

        # start the finetuning - either finetuning from scratch or resume from checkpoint
        if self.finetune_from_scratch:
            trainer.train()
        else:
            trainer.train(resume_from_checkpoint=True)

        # save the model to local directory
        trainer.save_state()
        trainer.save_model(self.saved_model_path)

    def __call__(self):
        return self.finetune()


# set up evaluation code
class Evaluation():
    def __init__(self, dev_pkl, test_pkl, processor_path, saved_model_path):
        self.dev_pkl = dev_pkl
        self.test_pkl = test_pkl
        self.processor_path = processor_path
        self.saved_model_path = saved_model_path

    # load the pickle data file - train data not required here
    def load_pickle_data(self):

        with open(self.dev_pkl, 'rb') as f:
            df_dev = pickle.load(f)
            
        with open(self.test_pkl, 'rb') as f:
            df_test = pickle.load(f)
            
        # make it into a DatasetDict Object
        dataset = DatasetDict({
            "dev": Dataset.from_pandas(df_dev),
            "test": Dataset.from_pandas(df_test)
        })

        return dataset

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

    # get the prediction result
    def map_to_result_gpu(self, batch, model, processor):
        model.to(device)
        with torch.no_grad():
            input_values = torch.tensor(batch["input_values"], device=device).unsqueeze(0)
            logits = model(input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_str"] = processor.batch_decode(pred_ids)[0]
        batch["text"] = processor.decode(batch["labels"], group_tokens=False)

        return batch

    # get prediction score
    def evaluate(self):

        # load the saved model and processor from local
        model = Wav2Vec2ForCTC.from_pretrained(self.saved_model_path)
        processor = Wav2Vec2Processor.from_pretrained(self.processor_path)
        
        # load the dev and test dataset
        dataset = self.load_pickle_data()

        # preprocess the dataset to feed into the transformer
        dataset = dataset.map(lambda x: self.preprocess_dataset_for_transformer(x, processor), remove_columns=dataset.column_names["dev"], num_proc=1)

        # get the prediction result
        results_dev = dataset["dev"].map(lambda x: self.map_to_result_gpu(x, model, processor), remove_columns=dataset["dev"].column_names)
        results_test = dataset["test"].map(lambda x: self.map_to_result_gpu(x, model, processor), remove_columns=dataset["test"].column_names)

        # define evaluation metric
        wer_metric = load_metric("wer")

        # get the wer of the dev and the test set
        print("Validation WER: {:.5f}".format(wer_metric.compute(predictions=results_dev["pred_str"], references=results_dev["text"])))
        print("Test WER: {:.5f}".format(wer_metric.compute(predictions=results_test["pred_str"], references=results_test["text"])))

    def __call__(self):
        return self.evaluate()

if __name__ == "__main__":
    
    ########## GET AUDIO LENGTH DISTRIBUTION ##########

    # print('Getting the audio length distribution of the train dataset\n')
    # distribution = FinetuningPreparation(train_pkl='./pkl/magister_data_v2_wav_16000_train.pkl',
    #                                      dev_pkl='./pkl/magister_data_v2_wav_16000_dev.pkl', 
    #                                      test_pkl='./pkl/magister_data_v2_wav_16000_test.pkl',
    #                                      processor_path='./processor/', 
    #                                      max_sample_length=None, 
    #                                      mode='get_audio_length_distribution')

    # distribution()

    ####################################################

    ########## GET THE PREPARED DATASET FOR THE FINETUNING TASK ##########

    # print('Getting the prepared dataset for the finetuning task\n')
    # data_preparation = FinetuningPreparation(train_pkl='./pkl/magister_data_v2_wav_16000_train.pkl',
    #                                      dev_pkl='./pkl/magister_data_v2_wav_16000_dev.pkl', 
    #                                      test_pkl='./pkl/magister_data_v2_wav_16000_test.pkl',
    #                                      processor_path='./processor/', 
    #                                      max_sample_length=450000, 
    #                                      mode='finetuning_prep')

    # dataset, _ = data_preparation()
    # print(dataset)

    ####################################################

    ########## FINETUNING THE MODEL - FROM SCRATCH ##########

    # finetune_model = Finetuning(train_pkl='./pkl/magister_data_v2_wav_16000_train.pkl', 
    #                             dev_pkl='./pkl/magister_data_v2_wav_16000_dev.pkl', 
    #                             test_pkl='./pkl/magister_data_v2_wav_16000_test.pkl', 
    #                             processor_path='./processor/', 
    #                             checkpoint_path='./ckpt/', 
    #                             pretrained_model_path='./wav2vec2_base_model/', 
    #                             saved_model_path='./saved_model/', 
    #                             max_sample_length=450000, 
    #                             batch_size=8, 
    #                             epochs=10, 
    #                             lr=1e-4, 
    #                             weight_decay=0.005, 
    #                             warmup_steps=1000, 
    #                             finetune_from_scratch=True)

    # finetune_model()

    ####################################################

    ########## FINETUNING THE MODEL - RESUMING FROM CHECKPOINT ##########

    # finetune_model = Finetuning(train_pkl='./pkl/magister_data_v2_wav_16000_train.pkl', 
    #                             dev_pkl='./pkl/magister_data_v2_wav_16000_dev.pkl', 
    #                             test_pkl='./pkl/magister_data_v2_wav_16000_test.pkl', 
    #                             processor_path='./processor/', 
    #                             checkpoint_path='./ckpt/', 
    #                             pretrained_model_path='./saved_model/', 
    #                             saved_model_path='./saved_model/', 
    #                             max_sample_length=450000, 
    #                             batch_size=8, 
    #                             epochs=15, 
    #                             lr=1e-4, 
    #                             weight_decay=0.005, 
    #                             warmup_steps=1000, 
    #                             finetune_from_scratch=False)

    # finetune_model()

    ####################################################

    ########## EVALUATION OF THE FINETUNED MODEL ##########
    
    evaluation = Evaluation(dev_pkl='./pkl/magister_data_v2_wav_16000_dev.pkl', 
                            test_pkl='./pkl/magister_data_v2_wav_16000_test.pkl', 
                            processor_path='./processor/', 
                            saved_model_path='./saved_model/')

    evaluation()

    ####################################################