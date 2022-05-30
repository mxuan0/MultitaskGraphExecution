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

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def gen_erdos_renyi_algo_results(rand_generator, num_graph, num_node, task_list, datasavefp='Data/', directed=False):
    gen_erdos_renyi(rand_generator, int(num_graph), int(num_node), datasavefp+"_", directed)
    src_nodes = torch.argmax(rand_generator.sample((int(num_graph), int(num_node))), dim=1)
    pdb.set_trace()
    gen_multi_algo_data(
        datasavefp+"_"+"erdosrenyi" + num_graph + "_"+ num_node + ".pt",
        src_nodes,
        task_list,
        True
    )

torch.manual_seed(0)

# task = 'bfs bf'
task = 'bf'
# algo_names = ['bfs', 'bf']
algo_names = ['bf']
# task_list = ['bfs', 'bf'] # or noalgo_?
task_list = ['bf'] # or noalgo_?

ngraph_train = '100'
ngraph_val = '100'
# ngraph_test = ['100', '100']
ngraph_test = ['200', '200', '200']

nnode = '20'
nnode_test = ['20', '50', '100']

logger = _logger()

# rand_gen = torch.distributions.Uniform(0.0, 1.0)
# gen_erdos_renyi_algo_results(rand_gen, ngraph_train, nnode, algo_names, 'Data/train')
# gen_erdos_renyi_algo_results(rand_gen, ngraph_val, nnode, algo_names, 'Data/val')
# for i in range(len(nnode_test)):
#     gen_erdos_renyi_algo_results(rand_gen, ngraph_test[i], nnode_test[i], algo_names, 'Data/test')




      
                    
device = 'cpu'
latentdim = 32
encdim = 32 
noisedim = 0
metadata = info(logger, algo_names)
model = arch.NeuralExecutor3_(device,
                              metadata['nodedim'],
                              metadata['edgedim'],
                              latentdim,
                              encdim,
                              algo_names,
                              pred=metadata['pred'],
                              noise=noisedim
                              )

train_params = {}
train_params['optimizer'] = 'sgd'
train_params['epochs'] = 10
train_params['lr'] = 1
train_params['warmup'] = 0
train_params['earlystop'] = True
train_params['patience'] = 1
train_params['weightdecay'] = 0.00001
train_params['schedpatience'] = 0
train_params['tempinit'] = 1.0
train_params['temprate'] = 1.0
train_params['tempmin'] = 1.0
train_params['earlytol'] = 5e-5
train_params['ksamples'] = 1
train_params['task'] = task
train_params['batchsize'] = 10

#for seq reptile 
train_params['K'] = 10
train_params['alpha'] = 0.0001

# train_stream, val_stream, test_stream = multi_stream(ngraph_train, ngraph_val, 
#                                                     nnode, logger, algo_names,
#                                                     ngraph_test, nnode_test)

# run_multi(model, logger, task_list, train_stream, val_stream, train_params, test_stream, device='cpu')

train_stream, val_stream, test_stream, test_datafp = seq_reptile_stream(ngraph_train, ngraph_val, nnode, logger, algo_names,
                 ngraph_test, nnode_test)

writer = run_seq_reptile(model, logger, task_list, train_stream, val_stream, train_params, test_stream, test_datafp, writer)
writer.close()