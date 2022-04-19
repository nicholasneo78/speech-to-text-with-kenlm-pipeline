from clearml import Task, Dataset
import yaml

# get task configs - ONLY THING NEEDED TO CHANGE
CONFIG_FILE = './config/config_task_finetuning_librispeech.yaml'

with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

# clearml configs
PROJECT_NAME = config['project_name']
TASK_NAME = config['task_name']
DATASET_NAME = config['dataset_name']
OUTPUT_URL = config['output_url']
DATASET_PROJECT = config['dataset_project']

task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME, output_uri=OUTPUT_URL)
task.set_base_docker(
    docker_image="nicholasneo78/w2v2_kenlm_pipeline:v0.2.3",
)

# get the args for data preprocessing
args = {
    'dataset_task_id': config['dataset_task_id'],
    'dataset_task_id_w2v2_base': config['dataset_task_id_w2v2_base'],
    'train_pkl': config['train_pkl'],
    'dev_pkl': config['dev_pkl'],
    'test_pkl': config['test_pkl'],
    'processor_path': config['processor_path'],
    'checkpoint_path': config['checkpoint_path'],
    'pretrained_model_path': config['pretrained_model_path'],
    'saved_model_path': config['saved_model_path'],
    'max_sample_length': config['max_sample_length'],
    'batch_size': config['batch_size'],
    'epochs': config['epochs'],
    'lr': config['lr'],
    'weight_decay': config['weight_decay'],
    'warmup_steps': config['warmup_steps'],
    'finetune_from_scratch': config['finetune_from_scratch']
}

task.connect(args)

# execute clearml
task.execute_remotely(queue_name=config['queue'], exit_process=True)

from preprocessing.finetuning import Finetuning

# register clearml dataset
# USE DATASET.GET INSTEAD AND SEE IF IT WORKS!

# obtain the pkl file
dataset = Dataset.create(
    dataset_project=DATASET_PROJECT,
    dataset_name=DATASET_NAME,
    parent_datasets=[args['dataset_task_id']]
)

# obtain the wav2vec2_base pretrained model
dataset_w2v2_base = Dataset.create(
    dataset_project=DATASET_PROJECT,
    dataset_name=DATASET_NAME,
    parent_datasets=[args['dataset_task_id_w2v2_base']]
)

# import dataset
dataset_path = dataset.get_local_copy()
dataset_w2v2_base_path = dataset_w2v2_base.get_local_copy()

# process
finetune_model = Finetuning(train_pkl=f'{args["train_pkl"]}', 
                                dev_pkl=f'{args["dev_pkl"]}', 
                                test_pkl=f'{args["test_pkl"]}', 
                                processor_path=args["processor_path"], 
                                checkpoint_path=args["checkpoint_path"], 
                                pretrained_model_path=f'{dataset_w2v2_base_path}/{args["pretrained_model_path"]}', 
                                saved_model_path=args["saved_model_path"], 
                                max_sample_length=args["max_sample_length"], 
                                batch_size=args["batch_size"], 
                                epochs=args["epochs"], 
                                lr=args["lr"], 
                                weight_decay=args["weight_decay"], 
                                warmup_steps=args["warmup_steps"], 
                                finetune_from_scratch=args["finetune_from_scratch"])

checkpoint_path, processor_path, pretrained_model_path, saved_model_path = finetune_model()

dataset.add_files(checkpoint_path)
dataset.add_files(processor_path)
dataset.add_files(pretrained_model_path)
dataset.add_files(saved_model_path)

dataset.upload(output_url=OUTPUT_URL)
dataset.finalize()

print('Done')
