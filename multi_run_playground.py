import numpy as np
import torch

import evaluate as ev
import model as arch
from baselines import run_seq_reptile, run_multi
from datagen.algorithms import info
from get_streams import seq_reptile_stream, multi_stream
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

task = 'bfs bf'
algo_names = ['bfs', 'bf']
task_list = ['bfs', 'bf']  # or noalgo_?

ngraph_train_list = [['100', '100'], [
    '1000', '1000']]
ngraph_val = '100'
ngraph_test = ['200', '200']

nnode = '20'
nnode_test = ['20', '50']

rand_gen = torch.distributions.Uniform(0.0, 1.0)
# gen_erdos_renyi_algo_results(rand_gen, ngraph_train, nnode, algo_names, 'Data/train')
# gen_erdos_renyi_algo_results(rand_gen, ngraph_val, nnode, algo_names, 'Data/val')
# for i in range(len(nnode_test)):
#     gen_erdos_renyi_algo_results(rand_gen, ngraph_test[i], nnode_test[i], algo_names, 'Data/test')


device = 'cuda'
latentdim = 16
encdim = 16
noisedim = 0

train_params = {}
train_params['optimizer'] = 'adam'
train_params['epochs'] = 300
train_params['lr'] = 1
train_params['warmup'] = 0
train_params['earlystop'] = False
train_params['patience'] = 1
train_params['weightdecay'] = 0.00001
train_params['schedpatience'] = 0
train_params['tempinit'] = 1.0
train_params['temprate'] = 1.0
train_params['tempmin'] = 1.0
train_params['earlytol'] = 1e-4
train_params['ksamples'] = 1
train_params['task'] = task
train_params['batchsize'] = 100
# train_params['setting'] = 'BFS->BF'

# for adaptive scheduling
train_params['exponent'] = 1.0
# for seq reptile
train_params['K'] = 9
train_params['alpha'] = 5e-4
for setting in ['BFS->BF', 'BF->BFS']:  # , '1BF+1BFS', 'Random']:
    train_params['setting'] = setting
    for ngraph_train in ngraph_train_list:
        mean_results = []
        steps = 10
        for i in range(steps):
            logger = _logger(logfile='Data/multi.log')
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

            train_stream, val_stream, test_stream = multi_stream(ngraph_train, ngraph_val, nnode, logger, algo_names,
                                                                 ngraph_test, nnode_test, 'erdosrenyi', train_params[
                                                                     'setting'],
                                                                 batchsize=train_params['batchsize'])

            res = run_multi(model, logger, task_list, train_stream, val_stream,
                            train_params, test_stream, device=device)

            mean_results.append(res)
        # print("HERE")
        results_arr = np.array(mean_results, dtype=object)
        average_arr = np.average(results_arr, axis=0)

        test_size = len(nnode_test)
        metrics = [m for t in task_list for m in ev.get_metrics(t)]
        metrics_len = len(metrics)

        average_arr = average_arr.reshape(len(task_list), test_size)

        for i in range(test_size):
            A = (average_arr[:, i])
            A = np.concatenate((A[0], A[1]))
            for ith, metric in enumerate(metrics):
                logger.info("average " + metric + ": {}".format(A[ith]))
                print("average " + metric + ": {}".format(A[ith]))
