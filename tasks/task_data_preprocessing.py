from clearml import Task, Dataset
import yaml

# get task configs - ONLY THING NEEDED TO CHANGE
CONFIG_FILE = './config/config_task_data_preprocessing_librispeech.yaml'

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
librispeech_train_pkl = GeneratePickleFromManifest(manifest_path=f'{dataset_path}/train/train_manifest.json', 
                                                   pkl_filename=f'{dataset_path}/pkl/librispeech_train.pkl', 
                                                   additional_preprocessing=args['additional_preprocessing'])

librispeech_dev_pkl = GeneratePickleFromManifest(manifest_path=f'{dataset_path}/dev/dev_manifest.json', 
                                                    pkl_filename=f'{dataset_path}/pkl/librispeech_dev.pkl', 
                                                    additional_preprocessing=args['additional_preprocessing'])

librispeech_test_pkl = GeneratePickleFromManifest(manifest_path=f'{dataset_path}/test/test_manifest.json', 
                                                    pkl_filename=f'{dataset_path}/pkl/librispeech_test.pkl', 
                                                    additional_preprocessing=args['additional_preprocessing'])

df_train = librispeech_train_pkl()
df_dev = librispeech_dev_pkl()
df_test = librispeech_test_pkl()

dataset.add_files(df_train)
dataset.add_files(df_dev)
dataset.add_files(df_test)

dataset.upload(output_url=OUTPUT_URL)
dataset.finalize()

print('Done')