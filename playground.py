from calendar import c
from datagen.algorithms import info, gen_multi_algo_data
from datagen.graphgen import gen_erdos_renyi, gen_barabasi_albert, gen_twod_grid
import torch, pdb, logging, sys
from loss import LossAssembler, create_loss_class
import model as arch
import initialisation as init
from train import train, train_metadata
import evaluate as ev
from torch.utils.data import DataLoader
from tqdm import tqdm
from baselines import run_multi, run_seq_reptile
from logger import _logger
from get_streams import multi_stream, seq_reptile_stream
torch.manual_seed(10)


task = 'bfs bf'
algo_names = ['bfs']
task_list = ['bfs'] # or noalgo_?

ngraph_train = '2000'
ngraph_val = '100'
ngraph_test = ['200', '200']

nnode = '20'
nnode_test = ['20', '50']

data_folder = 'Data'
graph = 'erdosrenyi'
log_file = 'multi.log'

# gen_erdos_renyi_algo_results(rand_gen, ngraph_train, nnode, algo_names, f'{data_folder}/train')
# gen_erdos_renyi_algo_results(rand_gen, ngraph_val, nnode, algo_names, f'{data_folder}/val')
# for i in range(len(nnode_test)):
#     gen_erdos_renyi_algo_results(rand_gen, ngraph_test[i], nnode_test[i], algo_names, 'Data/test')

      
                    
device = 'cpu'
latentdim = 32
encdim = 32 
noisedim = 0

train_params = {}
train_params['optimizer'] = 'adam'
train_params['epochs'] = 50
train_params['lr'] = 1e-4
train_params['warmup'] = 0
train_params['earlystop'] = False
train_params['patience'] = 2
train_params['weightdecay'] = 0.00001
train_params['schedpatience'] = 0
train_params['tempinit'] = 1.0
train_params['temprate'] = 1.0
train_params['tempmin'] = 1.0
train_params['earlytol'] = 1e-4
train_params['ksamples'] = 1
train_params['task'] = task
train_params['batchsize'] = 50

#for adaptive scheduling
train_params['exponent'] = 1.0

#for seq reptile 
train_params['K'] = 5
train_params['alpha'] = 1e-4


logger = _logger(logfile=f'{data_folder}/{log_file}')
metadata = info(logger, algo_names)
model = arch.NeuralExecutor3(device,
                              metadata['nodedim'],
                              metadata['edgedim'],
                              latentdim,
                              encdim,
                              pred=metadata['pred'],
                              noise=noisedim)
train_stream, val_stream, test_stream = multi_stream(ngraph_train, ngraph_val, 
                                                    nnode, logger, algo_names,
                                                    ngraph_test, nnode_test, graph=graph,
                                                    batchsize=train_params['batchsize'], data_folder=data_folder)
run_multi(model, logger, task_list, train_stream, val_stream, train_params, test_stream, device='cpu')


# logger = _logger(logfile='Data/seq_reptile.log')
# metadata = info(logger, algo_names)
# model = arch.NeuralExecutor3_(device,
#                               metadata['nodedim'],
#                               metadata['edgedim'],
#                               latentdim,
#                               encdim,
#                               algo_names,
#                               pred=metadata['pred'],
#                               noise=noisedim
#                               )

# train_stream, val_stream, test_stream = seq_reptile_stream(ngraph_train, ngraph_val, nnode, logger, algo_names,
#                  ngraph_test, nnode_test, batchsize=train_params['batchsize'])

# run_seq_reptile(model, logger, task_list, train_stream, val_stream, train_params, test_stream)