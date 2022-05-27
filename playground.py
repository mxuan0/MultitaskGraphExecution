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
from baselines import run_multi
from logger import _logger
from get_streams import multi_stream

# def gen_erdos_renyi_algo_results(rand_generator, num_graph, num_node, task_list, datasavefp='Data/', directed=False):
#     gen_erdos_renyi(rand_generator, int(num_graph), int(num_node), datasavefp+"_", directed)
#     src_nodes = torch.argmax(rand_generator.sample((int(num_graph), int(num_node))), dim=1)
#     pdb.set_trace()
#     gen_multi_algo_data(
#         datasavefp+"_"+"erdosrenyi" + num_graph + "_"+ num_node + ".pt",
#         src_nodes,
#         task_list,
#         True
#     )

rand_gen = torch.distributions.Uniform(0.0, 1.0)

task = 'bfs bf'

ngraph_train = '100'
ngraph_val = '30'
ngraph_test = ['100', '100']

algo_names = ['bfs', 'bf']
task_list = ['bfs', 'bf'] # or noalgo_?

nnode = '20'
nnode_test = ['20', '50']

logger = _logger()

# gen_erdos_renyi_algo_results(rand_gen, ngraph_train, nnode, algo_names, 'Data/train')
# gen_erdos_renyi_algo_results(rand_gen, ngraph_val, nnode, algo_names, 'Data/val')
# for i in range(len(nnode_test)):
#     gen_erdos_renyi_algo_results(rand_gen, ngraph_test[i], nnode_test[i], algo_names, 'Data/test')
                    
device = 'cpu'
latentdim = 32
encdim = 32 
noisedim = 0
metadata = info(logger, algo_names)
model = arch.NeuralExecutor3(device,
                              metadata['nodedim'],
                              metadata['edgedim'],
                              latentdim,
                              encdim,
                              pred=metadata['pred'],
                              noise=noisedim
                              )

train_params = {}
train_params['optimizer'] = 'adam'
train_params['epochs'] = 10
train_params['lr'] = 0.0001
train_params['warmup'] = 0
train_params['earlystop'] = True
train_params['patience'] = 1
train_params['weightdecay'] = 0.00001
train_params['schedpatience'] = 0
train_params['tempinit'] = 1.0
train_params['temprate'] = 1.0
train_params['tempmin'] = 1.0
train_params['earlytol'] = 5e-6
train_params['ksamples'] = 1
train_params['task'] = task
train_params['batchsize'] = 10

#for seq reptile 
train_params['K'] = 10
train_params['alpha'] = 1e-4

train_stream, val_stream, test_stream = multi_stream(ngraph_train, ngraph_val, 
                                                    nnode, logger, algo_names,
                                                    ngraph_test, nnode_test)

run_multi(model, logger, task_list, train_stream, val_stream, train_params, test_stream, device='cpu')