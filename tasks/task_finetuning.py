from clearml import Task, Dataset
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess data to generate pickle data files from the data manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments corresponding to the task initialisation
    parser.add_argument("--docker_image",                 type=str, help="the docker image used to load all the dependencies")
    parser.add_argument("--project_name",                 type=str, help="the clearml project name")
    parser.add_argument("--task_name",                    type=str, help="clearml task name")
    parser.add_argument("--dataset_name",                 type=str, help="name of the output dataset produced")
    parser.add_argument("--output_url",                   type=str, help="the clearml url that the task will be output at")
    parser.add_argument("--dataset_project",              type=str, help="the clearml path which the datasets resides")

    # arguments corresponding to getting data from different task id
    parser.add_argument("--dataset_pkl_task_id",          type=str, help="task id to retrieve the dataset")
    parser.add_argument("--dataset_pretrained_task_id",   type=str, help="task id to retrieve the pretrained/finetuned model")

    # arguments correspoding to the finetuning.py file
    parser.add_argument("--train_pkl",                    type=str, help="path to get the train pkl file")
    parser.add_argument("--dev_pkl",                      type=str, help="path to get the dev pkl file")
    parser.add_argument("--test_pkl",                     type=str, help="path to get the test pkl file")
    parser.add_argument("--input_processor_path",         type=str, help="path to retrieve the processor")
    parser.add_argument("--input_checkpoint_path",        type=str, help="path to retrieve the checkpoint")
    parser.add_argument("--input_pretrained_model_path",  type=str, help="path to retrieve the pretrained/finetuned model")
    parser.add_argument("--output_processor_path",        type=str, help="path to output the processor")
    parser.add_argument("--output_checkpoint_path",       type=str, help="path to output the checkpoint")
    parser.add_argument("--output_saved_model_path",      type=str, help="path to output the pretrained/finetuned model")
    parser.add_argument("--max_sample_length",            type=int, help="get the maximum sample length of the audio")
    parser.add_argument("--batch_size",                   type=int, help="batch size")
    parser.add_argument("--epochs",                       type=int, help="number of epochs")
    parser.add_argument("--lr",                           type=str, help="learning rate")
    parser.add_argument("--weight_decay",                 type=float, help="weight decay")
    parser.add_argument("--warmup_steps",                 type=int, help="number of steps for warmup - lower lr")
    parser.add_argument("--architecture",                 type=str, help="model based on wav2ve2 or wavlm")
    parser.add_argument("--finetune_from_scratch",        action='store_true', default=False, help="finetune model either from scratch or pre-existing finetuned model")

    # queue name
    parser.add_argument("--queue",                        type=str, help="the queue name for clearml")   

    return parser.parse_args(sys.argv[1:])

arg = parse_args()

# # clearml configs
# PROJECT_NAME = arg.project_name
# TASK_NAME = arg.task_name
# DATASET_NAME = arg.dataset_name
# OUTPUT_URL = arg.output_url
# DATASET_PROJECT = arg.dataset_project

task = Task.init(project_name=arg.project_name, task_name=arg.task_name, output_uri=arg.output_url)
task.set_base_docker(
    docker_image=arg.docker_image,
)

# # get the args for data preprocessing
# args = {
#     'dataset_pkl_task_id': arg.dataset_pkl_task_id,
#     'dataset_pretrained_task_id': arg.dataset_pretrained_task_id,

#     'train_pkl': arg.train_pkl,
#     'dev_pkl': arg.dev_pkl,
#     'test_pkl': arg.test_pkl,

#     'input_processor_path': arg.input_processor_path,
#     'input_checkpoint_path': arg.input_checkpoint_path,
#     'input_pretrained_model_path': arg.input_pretrained_model_path,

#     'output_processor_path': arg.output_processor_path,
#     'output_checkpoint_path': arg.output_checkpoint_path,
#     'output_saved_model_path': arg.output_saved_model_path,

#     'max_sample_length': arg.max_sample_length,
#     'batch_size': arg.batch_size,
#     'epochs': arg.epochs,
#     'lr': arg.lr,
#     'weight_decay': arg.weight_decay,
#     'warmup_steps': arg.warmup_steps,
#     'architecture': arg.architecture,
#     'finetune_from_scratch': arg.finetune_from_scratch
# }

# execute clearml
task.execute_remotely(queue_name=arg.queue, exit_process=True)

from preprocessing.finetuning import Finetuning

# obtaining the pkl file
dataset_pkl = Dataset.get(dataset_id=arg.dataset_pkl_task_id)
dataset_pkl_path = dataset_pkl.get_local_copy()

# obtaining the wav2vec2_base_pretrained model or resume finetuning
dataset_pretrained = Dataset.get(dataset_id=arg.dataset_pretrained_task_id)
dataset_pretrained_path = dataset_pretrained.get_local_copy()

# create new dataset object to store the checkpoint, processor, and saved model
dataset = Dataset.create(
    dataset_project=arg.dataset_project,
    dataset_name=arg.dataset_name,
)

# process
finetune_model = Finetuning(train_pkl=f'{dataset_pkl_path}/{arg.train_pkl}', 
                            dev_pkl=f'{dataset_pkl_path}/{arg.dev_pkl}', 
                            test_pkl=f'{dataset_pkl_path}/{arg.test_pkl}', 
                                
                            input_processor_path= f'{dataset_pretrained_path}/{arg.input_processor_path}', 
                            input_checkpoint_path= f'{dataset_pretrained_path}/{arg.input_checkpoint_path}', 
                            input_pretrained_model_path=f'{dataset_pretrained_path}/{arg.input_pretrained_model_path}',
                            
                            output_processor_path= arg.output_processor_path, 
                            output_checkpoint_path= arg.output_checkpoint_path, 
                            output_saved_model_path=arg.output_saved_model_path, 
                            
                            max_sample_length=arg.max_sample_length, 
                            batch_size=arg.batch_size, 
                            epochs=arg.epochs, 
                            lr=float(arg.lr), 
                            weight_decay=arg.weight_decay, 
                            warmup_steps=arg.warmup_steps, 
                            finetune_from_scratch=arg.finetune_from_scratch,
                            architecture=arg.architecture
                    )

checkpoint_path, processor_path, pretrained_model_path, saved_model_path = finetune_model()

dataset.add_files(path=checkpoint_path, local_base_folder='root/')
dataset.add_files(path=processor_path, local_base_folder='root/')
dataset.add_files(path=saved_model_path, local_base_folder='root/')

dataset.upload(output_url=arg.output_url)
dataset.finalize()

print('Done')
