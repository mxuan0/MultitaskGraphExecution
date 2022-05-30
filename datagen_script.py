import os 
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('data_folder')
parser.add_argument('data_mode')
args = parser.parse_args()

data_folder = args.data_folder
data_mode = args.data_mode

args = {
    'datagenonly': True,
    'normaliseloss': False,
    'nnodes': 50,
    'ngraphs': 10,
    'datagraphsetname': f'{data_folder}/{data_mode}',
    'graphtypes': 'BarabasiAlbert',
    'graphparams': '5',
    'task': "'bfs bf'",
    
}

if not os.path.isdir(data_folder):
    os.makedirs(data_folder)

strargs = ' '.join([f'--{key} {val}' for key, val in args.items()])


command = "python3 main.py " + strargs
print(command)

os.system(command)

