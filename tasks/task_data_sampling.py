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
    parser.add_argument("--input_train_dict",             type=dict, help="path to the input train pickle file")
    parser.add_argument("--input_dev_dict",               type=dict, help="path to the input dev pickle file")
    parser.add_argument("--input_test_dict",              type=dict, help="path to the input test pickle file")
    parser.add_argument("--output_pkl_train",             type=str, help="path to produce the sampled train pickle file")
    parser.add_argument("--output_pkl_dev",               type=str, help="path to produce the dev pickle file")
    parser.add_argument("--output_pkl_test",              type=str, help="path to produce the test pickle file")
    parser.add_argument("--sampling_mode",                type=str, help="sample either the manual way or the least way")
    parser.add_argument("--random_state",                 type=int, help="random seed")


    # queue name
    parser.add_argument("--queue",                        type=str, help="the queue name for clearml")   

    return parser.parse_args(sys.argv[1:])

arg = parse_args()

task = Task.init(project_name=arg.project_name, task_name=arg.task_name, output_uri=arg.output_url)
task.set_base_docker(
    docker_image=arg.docker_image,
)

# execute clearml
task.execute_remotely(queue_name=arg.queue, exit_process=True)

from preprocessing.data_sampling import DataSampling

# obtain the pkl files
dataset_pkl = Dataset.get(dataset_id=arg.dataset_task_id)
dataset_pkl_path = dataset_pkl.get_local_copy()

# register clearml sampled pkl file dataset
dataset = Dataset.create(
    dataset_project=arg.dataset_project,
    dataset_name=arg.dataset_name
)

# proceed with the sampling
pkl_train = DataSampling(data_dict=f'{dataset_pkl_path}/{arg.input_train_dict}',
                         sampling_mode=arg.sampling_mode, 
                         final_pkl_location=arg.output_pkl_train, 
                         random_state=arg.random_state)

pkl_dev = DataSampling(data_dict=f'{dataset_pkl_path}/{arg.input_dev_dict}',
                       sampling_mode=arg.sampling_mode, 
                       final_pkl_location=arg.output_pkl_dev, 
                       random_state=arg.random_state)

pkl_test = DataSampling(data_dict=f'{dataset_pkl_path}/{arg.input_test_dict}',
                        sampling_mode=arg.sampling_mode, 
                        final_pkl_location=arg.output_pkl_test, 
                        random_state=arg.random_state)

_, pkl_train_path = pkl_train()
_, pkl_dev_path = pkl_dev()
_, pkl_test_path = pkl_test()

# declare a root folder so that the subfolders can be retained by clearml
dataset.add_files(path=pkl_train_path, local_base_folder='root/')
dataset.add_files(path=pkl_dev_path, local_base_folder='root/')
dataset.add_files(path=pkl_test_path, local_base_folder='root/')

dataset.upload(output_url=arg.output_url)
dataset.finalize()

print('Done')


