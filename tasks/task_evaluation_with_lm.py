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
    parser.add_argument("--output_url",                   type=str, help="the clearml url that the task will be output at")

    # arguments corresponding to getting data from different task id
    parser.add_argument("--dataset_pkl_task_id",          type=str, help="task id to retrieve the dataset")
    parser.add_argument("--dataset_finetuned_task_id",    type=str, help="task id to retrieve the finetuned model")
    parser.add_argument("--lm_id",                        type=str, help="task id to retrieve the kenlm language model")
    
    # arguments corresponding to the evaluation_with_lm.py file
    parser.add_argument("--test_pkl",                     type=str, help="path to get the test pkl file")
    parser.add_argument("--finetuned_model_path",         type=str, help="path to get the finetuned model")
    parser.add_argument("--input_processor_path",         type=str, help="path to get the processor")
    parser.add_argument("--lm_path",                      type=str, help="path to get the language model")
    parser.add_argument("--alpha",                        type=float, help="alpha")
    parser.add_argument("--beta",                         type=float, help="beta")
    parser.add_argument("--architecture",                 type=str, help="model based on wav2ve2 or wavlm")

    # queue name
    parser.add_argument("--queue",                        type=str, help="the queue name for clearml")   

    return parser.parse_args(sys.argv[1:])

arg = parse_args()

# clearml configs
PROJECT_NAME = arg.project_name
TASK_NAME = arg.task_name
OUTPUT_URL = arg.output_url

task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME, output_uri=OUTPUT_URL)
task.set_base_docker(
    docker_image=arg.docker_image,
)

# get the args for data preprocessing
args = {
    'dataset_pkl_task_id': arg.dataset_pkl_task_id,
    'dataset_finetuned_task_id': arg.dataset_finetuned_task_id,
    'lm_id': arg.lm_id,

    'test_pkl': arg.test_pkl,
    'finetuned_model_path': arg.finetuned_model_path,
    'input_processor_path': arg.input_processor_path,
    'lm_path': arg.lm_path,

    'alpha': arg.alpha,
    'beta': arg.beta,
    'architecture': arg.architecture
}

# task.connect(args)

# execute clearml
task.execute_remotely(queue_name=arg.queue, exit_process=True)

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

print(f'alpha: {args["alpha"]}')
print(f'alpha: {arg.alpha}')
print(f'beta: {args["beta"]}')
print(f'beta: {arg.beta}')

evaluation = EvaluationWithLM(finetuned_model_path=f'{dataset_finetuned_path}/{args["finetuned_model_path"]}',
                              processor_path=f'{dataset_finetuned_path}/{args["input_processor_path"]}',
                              lm_path=f'{lm_path}/{args["lm_path"]}', 
                              test_data_path=f'{dataset_pkl_path}/{args["test_pkl"]}', 
                              alpha=float(args["alpha"]), 
                              beta=float(args["beta"]),
                              architecture=args["architecture"])

greedy, beam = evaluation()

print('Done')