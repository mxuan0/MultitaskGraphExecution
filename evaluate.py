import numpy as np
import torch

def get_metrics(task_name):
    if task_name == 'noalgo_bfs':
        return noalgo_bfs_metrics()
    elif task_name == 'noalgo_bf':
        return noalgo_bf_metrics()
    elif task_name == 'noalgo_prims':
        return noalgo_prims_metrics()
    elif task_name == 'noalgo_dijkstra':
        return noalgo_dijkstra_metrics()
    elif task_name == 'noalgo_dfs':
        return noalgo_dfs_metrics()
    elif task_name == 'noalgo_widest':
        return noalgo_widest_metrics()
    elif task_name == 'noalgo_widestpar':
        return noalgo_widestpar_metrics()
    elif task_name == 'noalgo_mostrelseq':
        return noalgo_mostrelseq_metrics()
    elif task_name == 'noalgo_mostrelpar':
        return noalgo_mostrelpar_metrics()
    elif task_name == 'bfs':
        return bfs_metrics()
    elif task_name == 'bf':
        return bf_metrics()
    elif task_name == 'prims':
        return prims_metrics()
    elif task_name == 'dijkstra':
        return dijkstra_metrics()
    elif task_name == 'dfs':
        return dfs_metrics()
    elif task_name == 'widest':
        return widest_metrics()
    elif task_name == 'widestpar':
        return widestpar_metrics()
    elif task_name == 'mostrelseq':
        return mostrelseq_metrics()
    elif task_name == 'mostrelpar':
        return mostrelpar_metrics()
    else:
        raise NotImplementedError

def noalgo_bfs_metrics():
    return ['BFS: reachability last step accuracy',
            ]

def noalgo_bf_metrics():
    return ['BellmanFord: last step mean squared error',
            'BellmanFord: predecessors last step accuracy',
            ]

def noalgo_widestpar_metrics():
    return ['Widest (parallel): last step mean squared error',
            'Widest (parallel): predecessors last step accuracy',
            ]

def bfs_metrics():
    return ['BFS: reachability mean step accuracy',
            'BFS: reachability last step accuracy',
            'BFS: termination accuracy',
            ]

def bf_metrics():
    return ['BellmanFord: mean squared error',
            'BellmanFord: last step mean squared error',
            'BellmanFord: predecessors mean step accuracy',
            'BellmanFord: predecessors last step accuracy',
            'Termination accuracy',
            ]

def widestpar_metrics():
    return ['Widest (parallel): mean squared error',
            'Widest (parallel): last step mean squared error',
            'Widest (parallel): predecessors mean step accuracy',
            'Widest (parallel): predecessors last step accuracy',
            'Termination accuracy',
            ]

def prims_metrics():
    return ['Prims: next MST node mean accuracy',
            'Prims: key mean accuracy',
            'Prims: state last accuracy',
            'Prims: key last accuracy',
            'Prims: predecessors MST mean accuracy',
            'Prims: predecessors MST last accuracy',
            'Termination accuracy',
            ]

def noalgo_prims_metrics():
    return ['Prims: state last accuracy',
            'Prims: key last accuracy',
            'Prims: predecessors MST last accuracy',
            ]

def dijkstra_metrics():
    return ['Dijkstra: next MST node mean accuracy',
            'Dijkstra: key mean accuracy',
            'Dijkstra: state last accuracy',
            'Dijkstra: key last accuracy',
            'Dijkstra: predecessors MST mean accuracy',
            'Dijkstra: predecessors MST last accuracy',
            'Termination accuracy',
            ]

def widest_metrics():
    return ['Widest path: next MST node mean accuracy',
            'Widest path: key mean accuracy',
            'Widest path: state last accuracy',
            'Widest path: key last accuracy',
            'Widest path: predecessors MST mean accuracy',
            'Widest path: predecessors MST last accuracy',
            'Termination accuracy',
            ]

def mostrelseq_metrics():
    return ['Most reliable path: next MST node mean accuracy',
            'Most reliable path: key mean accuracy',
            'Most reliable path: state last accuracy',
            'Most reliable path: key last accuracy',
            'Most reliable path: predecessors MST mean accuracy',
            'Most reliable path: predecessors MST last accuracy',
            'Termination accuracy',
            ]

def mostrelpar_metrics():
    return ['Most reliable path (parallel): mean squared error',
            'Most reliable path (parallel): last step mean squared error',
            'Most reliable path (parallel): predecessors mean step accuracy',
            'Most reliable path (parallel): predecessors last step accuracy',
            'Termination accuracy',
            ]

def noalgo_mostrelpar_metrics():
    return ['Most reliable path (parallel): last step mean squared error',
            'Most reliable path (parallel): predecessors last step accuracy',
            ]

def noalgo_mostrelseq_metrics():
    return ['Most reliable path: state last accuracy',
            'Most reliable path: key last accuracy',
            'Most reliable path: predecessors MST last accuracy',
            ]

def noalgo_widest_metrics():
    return ['Widest path: state last accuracy',
            'Widest path: key last accuracy',
            'Widest path: predecessors MST last accuracy',
            ]

def noalgo_dijkstra_metrics():
    return ['Dijkstra: state last accuracy',
            'Dijkstra: key last accuracy',
            'Dijkstra: predecessors MST last accuracy',
            ]

def dfs_metrics():
    return ['DFS: next MST node mean accuracy',
            'DFS: key mean accuracy',
            'DFS: state last accuracy',
            'DFS: key last accuracy',
            'DFS: predecessors MST mean accuracy',
            'DFS: predecessors MST last accuracy',
            'Termination accuracy',
            ]

def noalgo_dfs_metrics():
    return ['DFS: state last accuracy',
            'DFS: key last accuracy',
            'DFS: predecessors MST last accuracy',
            ]

def evaluate(logger, device, test_stream, model, loss_mod, metrics):
    """
    test_streams: list of datastreams, they expected to be an IterableDataset]
    batch_size: how many graphs to accumulate to a batch"""
    res = []
    with torch.no_grad():
        for stream in test_stream:
            logger.info(stream.dataset.name)
            ngraphs_total = len(stream.dataset)
            total_test_acc = [0 for _ in metrics]
            for batch in stream:
                batch_test_acc = loss_mod.test_loss(logger, device, model, batch)
                total_test_acc = [cum + btl for cum, btl in zip(total_test_acc, batch_test_acc)]
            mean_test_acc = [metric.detach().cpu()/ngraphs_total for metric in total_test_acc]
            res.append(mean_test_acc)
            for ith, metric in enumerate(metrics):
                logger.info(metric+": {}".format(mean_test_acc[ith]))
                print(metric + ": {}".format(mean_test_acc[ith]))

    return res

def evaluate_single_algo(logger, device, test_stream, model, loss_mod_dict, metrics):
    """
    test_streams: list of datastreams, they expected to be an IterableDataset]
    batch_size: how many graphs to accumulate to a batch"""
    res = []
    with torch.no_grad():
        for algo in test_stream:
            logger.info(algo)
            for stream in test_stream[algo]:
                logger.info(stream.dataset.name)
                ngraphs_total = len(stream.dataset)
                total_test_acc = [0 for _ in metrics[algo]]
                for batch in stream:
                    batch_test_acc = loss_mod_dict[algo].test_loss(logger, device, model, batch, algo)
                    total_test_acc = [cum + btl for cum, btl in zip(total_test_acc, batch_test_acc)]
                mean_test_acc = [metric.detach().cpu().item()/ngraphs_total for metric in total_test_acc]
                res.append(np.array(mean_test_acc))
                for ith, metric in enumerate(metrics[algo]):
                    print(metric+": {}".format(mean_test_acc[ith]))
                    logger.info(metric+": {}".format(mean_test_acc[ith]))

    return res