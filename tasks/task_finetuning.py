from clearml import Task, Dataset
import yaml

# get task configs - ONLY THING NEEDED TO CHANGE
CONFIG_FILE = './config/config_task_finetuning_librispeech_from_scratch.yaml'

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
    #'pretrained_model_path': config['pretrained_model_path'],
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

# obtaining the pkl file
dataset_pkl = Dataset.get(dataset_id=args['dataset_task_id'])
dataset_pkl_path = dataset_pkl.get_local_copy()

# obtaining the wav2vec2_base_pretrained model or resume finetuning
dataset_w2v2_base = Dataset.get(dataset_id=args['dataset_task_id_w2v2_base'])
dataset_w2v2_base_path = dataset_w2v2_base.get_local_copy()

# create new dataset object to store the checkpoint, processor, and saved model
dataset = Dataset.create(
    dataset_project=DATASET_PROJECT,
    dataset_name=DATASET_NAME,
    # parent_datasets=[args['dataset_task_id']]
)


# process
finetune_model = Finetuning(train_pkl=f'{dataset_pkl_path}/{args["train_pkl"]}', 
                                dev_pkl=f'{dataset_pkl_path}/{args["dev_pkl"]}', 
                                test_pkl=f'{dataset_pkl_path}/{args["test_pkl"]}', 
                                processor_path=args["processor_path"], 
                                checkpoint_path=args["checkpoint_path"], 
                                pretrained_model_path=f'{dataset_w2v2_base_path}', 
                                saved_model_path=args["saved_model_path"], 
                                max_sample_length=args["max_sample_length"], 
                                batch_size=args["batch_size"], 
                                epochs=args["epochs"], 
                                lr=float(args["lr"]), 
                                weight_decay=args["weight_decay"], 
                                warmup_steps=args["warmup_steps"], 
                                finetune_from_scratch=args["finetune_from_scratch"])

checkpoint_path, processor_path, pretrained_model_path, saved_model_path = finetune_model()

dataset.add_files(path=checkpoint_path, local_base_folder='root/')
dataset.add_files(path=processor_path, local_base_folder='root/')
dataset.add_files(path=saved_model_path, local_base_folder='root/')

dataset.upload(output_url=OUTPUT_URL)
dataset.finalize()

print('Done')
