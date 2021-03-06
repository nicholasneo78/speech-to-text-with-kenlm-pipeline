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
    
    # arguments correspoding to the data_preprocessing.py file
    parser.add_argument("--dataset_task_id",              type=str, help="task id to retrieve the dataset")
    parser.add_argument("--manifest_path_train",          type=str, help="path to retrieve the train manifest")
    parser.add_argument("--pkl_train",                    type=str, help="path to produce the train pickle file")
    parser.add_argument("--manifest_path_dev",            type=str, help="path to retrieve the dev manifest")
    parser.add_argument("--pkl_dev",                      type=str, help="path to produce the dev pickle file")
    parser.add_argument("--manifest_path_test",           type=str, help="path to retrieve the test manifest")
    parser.add_argument("--pkl_test",                     type=str, help="path to produce the test pickle file")
    parser.add_argument("--label",                        type=str, help="data label")
    parser.add_argument("--additional_preprocessing",     type=str, help="any other special cases of text preprocessing needed based on the annotations")

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
#     'dataset_task_id': arg.dataset_task_id,
#     'manifest_path_train': arg.manifest_path_train,
#     'pkl_train': arg.pkl_train,
#     'manifest_path_dev': arg.manifest_path_dev,
#     'pkl_dev': arg.pkl_dev,
#     'manifest_path_test': arg.manifest_path_test,
#     'pkl_test': arg.pkl_test,
#     'additional_preprocessing': arg.additional_preprocessing
# }

# execute clearml
task.execute_remotely(queue_name=arg.queue, exit_process=True)

from preprocessing.data_preprocessing import GeneratePickleFromManifest

# obtaining the manifest file
dataset_manifest = Dataset.get(dataset_id=arg.dataset_task_id)
dataset_manifest_path = dataset_manifest.get_local_copy()

# register clearml pkl file dataset
dataset = Dataset.create(
    dataset_project=arg.dataset_project,
    dataset_name=arg.dataset_name,
    #parent_datasets=[arg.dataset_task_id]
)

# process
librispeech_train_pkl = GeneratePickleFromManifest(manifest_path=f'{dataset_manifest_path}/{arg.manifest_path_train}', 
                                                   pkl_filename=arg.pkl_train, 
                                                   label=arg.label,
                                                   additional_preprocessing=arg.additional_preprocessing)

librispeech_dev_pkl = GeneratePickleFromManifest(manifest_path=f'{dataset_manifest_path}/{arg.manifest_path_dev}', 
                                                 pkl_filename=arg.pkl_dev, 
                                                 label=arg.label,
                                                 additional_preprocessing=arg.additional_preprocessing)

librispeech_test_pkl = GeneratePickleFromManifest(manifest_path=f'{dataset_manifest_path}/{arg.manifest_path_test}', 
                                                  pkl_filename=arg.pkl_test, 
                                                  label=arg.label,
                                                  additional_preprocessing=arg.additional_preprocessing)

df_train, path_train = librispeech_train_pkl()
df_dev, path_dev = librispeech_dev_pkl()
df_test, path_test = librispeech_test_pkl()

# declare a root folder so that the subfolders can be retained by clearml
dataset.add_files(path=path_train, local_base_folder='root/')
dataset.add_files(path=path_dev, local_base_folder='root/')
dataset.add_files(path=path_test, local_base_folder='root/')

dataset.upload(output_url=arg.output_url)
dataset.finalize()

print('Done')