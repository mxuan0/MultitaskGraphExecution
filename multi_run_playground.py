import numpy as np
import torch

import evaluate as ev
import model as arch
from baselines import run_seq_reptile
from datagen.algorithms import info
from get_streams import seq_reptile_stream
from logger import _logger

torch.manual_seed(0)

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

task = 'bf'
algo_names = ['bf']
task_list = ['bf']  # or noalgo_?

ngraph_train = ['1000']
ngraph_val = '100'
ngraph_test = ['200', '200']

nnode = '20'
nnode_test = ['20', '50']

rand_gen = torch.distributions.Uniform(0.0, 1.0)
# gen_erdos_renyi_algo_results(rand_gen, ngraph_train, nnode, algo_names, 'Data/train')
# gen_erdos_renyi_algo_results(rand_gen, ngraph_val, nnode, algo_names, 'Data/val')
# for i in range(len(nnode_test)):
#     gen_erdos_renyi_algo_results(rand_gen, ngraph_test[i], nnode_test[i], algo_names, 'Data/test')


device = 'cpu'
latentdim = 32
encdim = 32
noisedim = 0

train_params = {}
train_params['optimizer'] = 'adam'
train_params['epochs'] = 300
train_params['lr'] = 1
train_params['warmup'] = 0
train_params['earlystop'] = False
train_params['patience'] = 2
train_params['weightdecay'] = 5e-5
train_params['schedpatience'] = 0
train_params['tempinit'] = 1.0
train_params['temprate'] = 1.0
train_params['tempmin'] = 1.0
train_params['earlytol'] = 1e-4
train_params['ksamples'] = 1
train_params['task'] = task
train_params['batchsize'] = 50

# for adaptive scheduling
train_params['exponent'] = 1.0

# for seq reptile
train_params['K'] = 10
train_params['alpha'] = 5e-4

# logger = _logger(logfile='Data/multi.log')
# metadata = info(logger, algo_names)
# model = arch.NeuralExecutor3(device,
#                               metadata['nodedim'],
#                               metadata['edgedim'],
#                               latentdim,
#                               encdim,
#                               pred=metadata['pred'],
#                               noise=noisedim
#                               )

# train_stream, val_stream, test_stream = multi_stream(ngraph_train, ngraph_val,
#                                                     nnode, logger, algo_names,
#                                                     ngraph_test, nnode_test,
#                                                     batchsize=train_params['batchsize'])

# run_multi(model, logger, task_list, train_stream, val_stream, train_params, test_stream, device='cpu')
mean_results = []
steps = 10
for i in range(steps):
    logger = _logger(logfile='Data/seq_reptile.log')
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

    train_stream, val_stream, test_stream = seq_reptile_stream(ngraph_train, ngraph_val, nnode, logger, algo_names,
                                                               ngraph_test, nnode_test,
                                                               batchsize=train_params['batchsize'])

    res = run_seq_reptile(model, logger, task_list, train_stream, val_stream, train_params, test_stream)
    mean_results.append(res)

results_arr = np.array(mean_results)
average_arr = np.average(results_arr, axis=0)

test_size = len(nnode_test)
metrics = [m for t in task_list for m in ev.get_metrics(t)]
metrics_len = len(metrics)

for i in range(test_size):
    # print("averaging on testset " + i)
    for ith, metric in enumerate(metrics):
        logger.info("average " + metric + ": {}".format(average_arr[i][ith]))
        print("average " + metric + ": {}".format(average_arr[i][ith]))
