from clearml import Task, Dataset
import yaml

# get task configs - ONLY THING NEEDED TO CHANGE
CONFIG_FILE = './config/task_build_lm/librispeech.yaml'

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
    'dataset_pkl_task_id': config["dataset_pkl_task_id"],
    'script_task_id': config["script_task_id"],
    'train_pkl': config["train_pkl"],
    'dev_pkl': config["dev_pkl"],
    'script_path': config["script_path"],
    'root_path': config["root_path"],
    'txt_filepath': config["txt_filepath"],
    'n_grams': config["n_grams"],
    'dataset_name_': config["dataset_name_"]
}

task.connect(args)

# execute clearml
task.execute_remotely(queue_name=config['queue'], exit_process=True)

from preprocessing.build_lm import BuildLM

# obtaining the pkl file
dataset_pkl = Dataset.get(dataset_id=args['dataset_pkl_task_id'])
dataset_pkl_path = dataset_pkl.get_local_copy()

# obtain the build_lm script path
get_script = Dataset.get(dataset_id=args['script_task_id'])
get_script_dir = get_script.get_local_copy()

# create new dataset object to store the language model
dataset = Dataset.create(
    dataset_project=DATASET_PROJECT,
    dataset_name=DATASET_NAME,
)

print(f'Script Path: {get_script_dir}')

get_lm = BuildLM(df_train_filepath=f'{dataset_pkl_path}/{args["train_pkl"]}',
                     df_dev_filepath=f'{dataset_pkl_path}/{args["dev_pkl"]}', 
                     script_path=f'{get_script_dir}/{args["script_path"]}', 
                     root_path=args["root_path"], #"/workspace", 
                     txt_filepath=args["txt_filepath"], 
                     n_grams=args["n_grams"], 
                     dataset_name=args["dataset_name_"])

lm_path = get_lm()

print('\nClearML add files now\n')

dataset.add_files(path=lm_path, local_base_folder='root/')

print('\n\nPassed\n\n')

dataset.upload(output_url=OUTPUT_URL)
dataset.finalize()

print('Done')