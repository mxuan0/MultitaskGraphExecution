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

torch.manual_seed(0)

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

def _logger(debug=False, logfile='Data/log.txt', verbose=False):
    logFormatter = logging.Formatter(
        '%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s'
    )
    rootLogger = logging.getLogger()
    # set debugging level
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    rootLogger.setLevel(level)

    # file logging to logfile
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # logging to std out
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    # if verbose is set we print level else we print warnings only
    if verbose:
        consoleHandler.setLevel(level)
    else:
        consoleHandler.setLevel(logging.WARNING)
    rootLogger.addHandler(consoleHandler)
    return rootLogger

rand_gen = torch.distributions.Uniform(0.0, 1.0)

# task = 'bfs bf'
task = 'bf'

ngraph_train = '100'
ngraph_val = '100'
ngraph_test = ['200', '200', '200']

# algo_names = ['bfs', 'bf']
algo_names = ['bf']

# task_list = ['bfs', 'bf']
task_list = ['bf']

nnode = '20'
nnode_test = ['20', '50', '100']

# gen_erdos_renyi_algo_results(rand_gen, ngraph_train, nnode, algo_names, 'Data/train')
# gen_erdos_renyi_algo_results(rand_gen, ngraph_val, nnode, algo_names, 'Data/val')
# for i in range(len(nnode_test)):
#     gen_erdos_renyi_algo_results(rand_gen, ngraph_test[i], nnode_test[i], algo_names, 'Data/test')

train_datafp = 'Data/train_erdosrenyi%s_%s' % (ngraph_train, nnode)
val_datafp = 'Data/val_erdosrenyi%s_%s' % (ngraph_val, nnode)
test_datafp = ['Data/test_erdosrenyi%s_%s' % (ngraph_test[i], nnode_test[i]) for i in range(len(nnode_test))]
batchsize = 10

import dataloaders as dl
dset = dl.MultiAlgo
#dset(None,datafp.split(' '),['bf', 'widest'],"Train")

logger = _logger()

train_stream = DataLoader(dset(logger,train_datafp.split(' '),algo_names,"Train"),
                             shuffle = True,
                             batch_size = batchsize,
                             collate_fn = dl.collate_multi_algo,
                             drop_last = False
                             )

val_stream = DataLoader(dset(logger,val_datafp.split(' '),algo_names,"Validation"),
                             shuffle = False,
                             batch_size = batchsize,
                             collate_fn = dl.collate_multi_algo,
                             drop_last = False
                             )
                             
test_stream = []
for fp in test_datafp:
    test_stream.append(DataLoader(dset(logger,[fp],algo_names,'Test'),
                                    shuffle = False,
                                    batch_size = batchsize,
                                    collate_fn = dl.collate_multi_algo,
                                    drop_last = False
                                    )
                        )
                    
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
train_params['batchsize'] = batchsize

algos = []
for task in task_list:
    algos.append(
        create_loss_class(task, {'hidekeys':True})
    )
loss_module = LossAssembler(device, logger, algos, {'ksamples':1})
recorder = None
# creating the training parameter dict
model_state, val_loss = train(logger,
                                device,
                                train_stream,
                                val_stream,
                                model,
                                train_params,
                                loss_module,
                                recorder
                                )

metrics = [m for t in task_list for m in ev.get_metrics(t)]
ev.evaluate(logger,
            device,
            test_stream,
            model,
            loss_module,
            metrics
            )