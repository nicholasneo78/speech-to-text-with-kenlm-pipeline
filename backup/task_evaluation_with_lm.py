from clearml import Task, Dataset
import yaml

# get task configs - ONLY THING NEEDED TO CHANGE
CONFIG_FILE = './config/task_evaluation_with_lm/librispeech_wav2vec2.yaml'
# CONFIG_FILE = './config/task_evaluation_with_lm/librispeech_wavlm.yaml'

with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

# clearml configs
PROJECT_NAME = config['project_name']
TASK_NAME = config['task_name']
OUTPUT_URL = config['output_url']


task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME, output_uri=OUTPUT_URL)
task.set_base_docker(
    docker_image="nicholasneo78/stt_with_kenlm_pipeline:v0.1.1"
)

# get the args for data preprocessing
args = {
    'dataset_pkl_task_id': config["dataset_pkl_task_id"],
    'dataset_finetuned_task_id': config["dataset_finetuned_task_id"],
    'lm_id': config["lm_id"],

    'test_pkl': config["test_pkl"],
    'finetuned_model_path': config["finetuned_model_path"],
    'input_processor_path': config["input_processor_path"],
    'lm_path': config["lm_path"],

    'alpha': config["alpha"],
    'beta': config["beta"],
    'architecture': config["architecture"]
}

task.connect(args)

# execute clearml
task.execute_remotely(queue_name=config['queue'], exit_process=True)

from preprocessing.evaluation_with_lm import EvaluationWithLM

# obtaining the pkl file
dataset_pkl = Dataset.get(dataset_id=args['dataset_pkl_task_id'])
dataset_pkl_path = dataset_pkl.get_local_copy()

# obtaining the finetuned model
dataset_finetuned = Dataset.get(dataset_id=args['dataset_finetuned_task_id'])
dataset_finetuned_path = dataset_finetuned.get_local_copy()

# obtaining the kenlm lm
lm = Dataset.get(dataset_id=args['lm_id'])
lm_path = lm.get_local_copy()


evaluation = EvaluationWithLM(finetuned_model_path=f'{dataset_finetuned_path}/{args["finetuned_model_path"]}',
                              processor_path=f'{dataset_finetuned_path}/{args["input_processor_path"]}',
                              lm_path=f'{lm_path}/{args["lm_path"]}', 
                              test_data_path=f'{dataset_pkl_path}/{args["test_pkl"]}', 
                              alpha=float(args["alpha"]), 
                              beta=float(args["beta"]),
                              architecture=args["architecture"])

greedy, beam = evaluation()

print('Done')