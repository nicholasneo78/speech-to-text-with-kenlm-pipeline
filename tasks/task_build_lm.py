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
    
    # arguments correspoding to getting data from different task id
    parser.add_argument("--dataset_pkl_task_id",          type=str, help="task id to retrieve the dataset")
    parser.add_argument("--script_task_id",               type=str, help="task id to get the build_lm.sh script")
    parser.add_argument("--kenlm_id",                     type=str, help="kenlm path to build the lm")

    # arguments correspoding to the build_lm.py file
    parser.add_argument("--train_pkl",                    type=str, help="path to get the train pkl file")
    parser.add_argument("--dev_pkl",                      type=str, help="path to get the dev pkl file")
    parser.add_argument("--script_path",                  type=str, help="filename of building the kenlm language model")
    parser.add_argument("--txt_filepath",                 type=str, help="path to get the text file with all the words from train and dev set")
    parser.add_argument("--n_grams",                      type=str, help="number of grams for the language model")
    parser.add_argument("--dataset_name_",                type=str, help="the name of the dataset")

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
#     'script_task_id': arg.script_task_id,
#     'kenlm_id': arg.kenlm_id,
#     'train_pkl': arg.train_pkl,
#     'dev_pkl': arg.dev_pkl,
#     'script_path': arg.script_path,
#     'txt_filepath': arg.txt_filepath,
#     'n_grams': arg.n_grams,
#     'dataset_name_': arg.dataset_name_
# }

# execute clearml
task.execute_remotely(queue_name=arg.queue, exit_process=True)

from preprocessing.build_lm import BuildLM

# obtaining the pkl file
dataset_pkl = Dataset.get(dataset_id=arg.dataset_pkl_task_id)
dataset_pkl_path = dataset_pkl.get_local_copy()

# obtain the build_lm script path
get_script = Dataset.get(dataset_id=arg.script_task_id)
get_script_dir = get_script.get_local_copy()

# get the kenlm build
get_kenlm = Dataset.get(dataset_id=arg.kenlm_id)
get_kenlm_dir = get_kenlm.get_local_copy()

# create new dataset object to store the language model
dataset = Dataset.create(
    dataset_project=arg.dataset_project,
    dataset_name=arg.dataset_name,
)

get_lm = BuildLM(df_train_filepath=f'{dataset_pkl_path}/{arg.train_pkl}',
                     df_dev_filepath=f'{dataset_pkl_path}/{arg.dev_pkl}', 
                     script_path=f'{get_script_dir}/{arg.script_path}', 
                     root_path=f'{get_kenlm_dir}', #"/workspace", 
                     txt_filepath=arg.txt_filepath, 
                     n_grams=arg.n_grams, 
                     dataset_name=arg.dataset_name_)

lm_path = get_lm()

dataset.add_files(path=lm_path, local_base_folder='root/')

dataset.upload(output_url=arg.output_url)
dataset.finalize()

print('Done')