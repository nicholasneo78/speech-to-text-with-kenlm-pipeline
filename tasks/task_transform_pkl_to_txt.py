from clearml import Task, Dataset
import yaml

# get task configs - ONLY THING NEEDED TO CHANGE
CONFIG_FILE = './config/task_transform_pkl_to_txt/librispeech.yaml'

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
    docker_image="nicholasneo78/stt_with_kenlm_pipeline:v0.1.1"
)

# get the args for data preprocessing
args = {
    'train_pkl': config["train_pkl"],
    'dev_pkl': config["dev_pkl"],
    'txt_filepath': config["txt_filepath"]
}

task.connect(args)

# execute clearml
task.execute_remotely(queue_name=config['queue'], exit_process=True)

from preprocessing.transform_pkl_to_txt import GetTxtFromPkl

# obtaining the pkl file
dataset_pkl = Dataset.get(dataset_id=args['dataset_pkl_task_id'])
dataset_pkl_path = dataset_pkl.get_local_copy()

# create new dataset object to store the checkpoint, processor, and saved model
dataset = Dataset.create(
    dataset_project=DATASET_PROJECT,
    dataset_name=DATASET_NAME,
)

get_txt_from_pkl = GetTxtFromPkl(df_train_filepath=f'{dataset_pkl_path}/{args["train_pkl"]}',
                                 df_dev_filepath=f'{dataset_pkl_path}/{args["dev_pkl"]}',
                                 txt_filepath=args["txt_filepath"])

text_path = get_txt_from_pkl()

dataset.add_files(path=text_path, local_base_folder='root/')

dataset.upload(output_url=OUTPUT_URL)
dataset.finalize()

print('Done')