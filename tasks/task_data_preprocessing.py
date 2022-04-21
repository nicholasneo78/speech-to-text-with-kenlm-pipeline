from clearml import Task, Dataset
import yaml

# get task configs - ONLY THING NEEDED TO CHANGE
CONFIG_FILE = './config/task_data_preprocessing/librispeech.yaml'

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
    docker_image="nicholasneo78/stt_with_kenlm_pipeline:v0.1.0",
)

# get the args for data preprocessing
args = {
    'dataset_task_id': config['dataset_task_id'],
    'manifest_path_train': config['manifest_path_train'],
    'pkl_train': config['pkl_train'],
    'manifest_path_dev': config['manifest_path_dev'],
    'pkl_dev': config['pkl_dev'],
    'manifest_path_test': config['manifest_path_test'],
    'pkl_test': config['pkl_test'],
    'additional_preprocessing': config['additional_preprocessing']
}

task.connect(args)

# execute clearml
task.execute_remotely(queue_name=config['queue'], exit_process=True)

from preprocessing.data_preprocessing import GeneratePickleFromManifest

# register clearml dataset
dataset = Dataset.create(
    dataset_project=DATASET_PROJECT,
    dataset_name=DATASET_NAME,
    parent_datasets=[args['dataset_task_id']]
)

# import dataset
dataset_path = dataset.get_local_copy()

# process
librispeech_train_pkl = GeneratePickleFromManifest(manifest_path=f'{dataset_path}/{args["manifest_path_train"]}', 
                                                   pkl_filename=args["pkl_train"], 
                                                   additional_preprocessing=args['additional_preprocessing'])

librispeech_dev_pkl = GeneratePickleFromManifest(manifest_path=f'{dataset_path}/{args["manifest_path_dev"]}', 
                                                 pkl_filename=args["pkl_dev"], 
                                                 additional_preprocessing=args['additional_preprocessing'])

librispeech_test_pkl = GeneratePickleFromManifest(manifest_path=f'{dataset_path}/{args["manifest_path_test"]}', 
                                                  pkl_filename=args["pkl_test"], 
                                                  additional_preprocessing=args['additional_preprocessing'])

df_train, path_train = librispeech_train_pkl()
df_dev, path_dev = librispeech_dev_pkl()
df_test, path_test = librispeech_test_pkl()

print(path_train)
print(path_dev)
print(path_test)

# declare a root folder so that the subfolders can be retained by clearml


dataset.add_files(path=path_train, local_base_folder='root/')
dataset.add_files(path=path_dev, local_base_folder='root/')
dataset.add_files(path=path_test, local_base_folder='root/')

dataset.upload(output_url=OUTPUT_URL)
dataset.finalize()

print('Done')