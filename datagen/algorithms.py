import pickle as pkl

import numpy as np
import torch
import pdb
def info(logger, algos):
    metadata = dict()
    metadata['nodedim'] = []
    metadata['edgedim'] = []
    metadata['pred'] = []
    for alg in algos:
        if alg == 'bfs':
            metadata['nodedim'] += [1]
            metadata['edgedim'] += [0]
            metadata['pred'] += [0]
        elif alg == 'bf':
            metadata['nodedim'] += [1]
            metadata['edgedim'] += [1]
            metadata['pred'] += [1]
        elif alg == 'prims':
            metadata['nodedim'] += [2]
            metadata['edgedim'] += [1]
            metadata['pred'] += [1]
        elif alg == 'dijkstra':
            metadata['nodedim'] += [2]
            metadata['edgedim'] += [1]
            metadata['pred'] += [1]
        elif alg == 'mostrelseq':
            metadata['nodedim'] += [2]
            metadata['edgedim'] += [1]
            metadata['pred'] += [1]
        elif alg == 'mostrelpar':
            metadata['nodedim'] += [1]
            metadata['edgedim'] += [1]
            metadata['pred'] += [1]
        elif alg == 'dfs':
            metadata['nodedim'] += [2]
            metadata['edgedim'] += [0]
            metadata['pred'] += [1]
        elif alg == 'widest':
            metadata['nodedim'] += [2]
            metadata['edgedim'] += [1]
            metadata['pred'] += [1]
        elif alg == 'widestpar':
            metadata['nodedim'] += [1]
            metadata['edgedim'] += [1]
            metadata['pred'] += [1]
        else:
            logger.exception('The algorithm task submitted does not exist: ' + alg)
    return metadata

def gen_info(algo):
    if algo == 'bf':
        return bellmanford_alg, bellmanford_store
    elif algo == 'bfs':
        return breadthfirstsearch_alg, breadthfirstsearch_store
    elif algo == 'prims':
        return prims_alg, prims_store
    elif algo == 'dijkstra':
        return dijkstra_alg, dijkstra_store
    elif algo == 'dfs':
        return dfs_alg, dfs_store
    elif algo == 'mostrelseq':
        return mostrelseq_alg, mostrelseq_store
    elif algo == 'mostrelpar':
        return mostrelpar_alg, mostrelpar_store
    elif algo == 'widest':
      return widest_alg, widest_store
    elif algo == 'widestpar':
      return widestpar_alg, widestpar_store
    else:
        raise NotImplementedError

### function to run several algos on the same graphs ( for multi-task )
def gen_multi_algo_data(graphfp, src_nodes, algos, hide_keys):
    uniform_gen = torch.distributions.Uniform(0.2, 1.0)

    adj = torch.load(graphfp)
    shape = adj.shape
    adj = adj_src_connect(adj,src_nodes)

    rand_mats = torch.tril(uniform_gen.sample(shape))
    weights = (rand_mats + rand_mats.transpose(1,2)) * adj

    for algo in algos:
        name_split = algo.split('_')
        algo = name_split[-1]
        tf = False if len(name_split) > 1 else True
        alg, store = gen_info(algo)
        steps = alg(adj, weights, src_nodes, hide_keys and tf)
        store(steps, graphfp[:-3]+f'_{algo}.pkl')

    return

### The algorithms here are designed for relatively small graphs as we rely on
### all the graph's adjacency matrices fitting into memory.
### Tasks are: Shortest path, Reachability

## Bellman-Ford (Shortest path)
def bellmanford_step(adj, weights, x, p):
    x_neigh = x.unsqueeze(1).expand_as(adj).transpose(1,2).masked_fill(~adj, adj.shape[1]+1)
    weights = weights.masked_fill(weights == 0, adj.shape[1]+1)
    x_next = torch.min(x, torch.min(x_neigh + weights, dim=1)[0])
    p_next = torch.where((x_next == (adj.shape[1]+1)) + (x_next == 0),
                       # torch.arange(0,adj.shape[1]).unsqueeze(0).expand_as(x_next).long(),
                       p,
                       torch.argmin(x_neigh + weights, dim=1).float()
                         ).float()
    return x_next, p_next

def bellmanford_alg(adj, weights, src, hide_keys):
    steps = []
    adj = (adj != 0)

    # we treat the number of nodes + 1 as infinity
    x_init = torch.zeros((adj.shape[0], adj.shape[1])).fill_(adj.shape[1]+1)
    x_init.scatter_(1, src.unsqueeze(1), 0)
    p_init = torch.where(x_init == 0,
                          torch.arange(0,adj.shape[1]).unsqueeze(0).expand_as(x_init).double(),
                          float('inf')
                       ).float()

    steps.append((x_init, torch.tensor([]), torch.zeros(adj.shape[0])))

    x_prev = x_init
    p_prev = p_init
    for i in range(adj.shape[1]):
        x, p = bellmanford_step(adj, weights, x_prev, p_prev)
        term = (((x-x_prev) == 0).all(1)).long()
        steps.append((x,p, term))
        x_prev = x
        p_prev = p
    longest_short_dist = torch.max(x, dim=-1)[0]
    for t, _, _ in steps:
        t.masked_scatter_(t == (adj.shape[1]+1),
                          (longest_short_dist+1).unsqueeze(-1).expand_as(t)[t == (adj.shape[1]+1)])
    steps.append((adj,weights,torch.tensor([])))
    return steps

def bellmanford_store(steps, store_fp):
    steps_numpy = [(t0.numpy(), t1.numpy(), t2.numpy()) for t0, t1, t2 in steps]

    with open(store_fp, 'wb') as f:
        pkl.dump(steps_numpy, f, pkl.HIGHEST_PROTOCOL)

    return

def gen_bellmanford_data(graphfp, src_nodes):
    uniform_gen = torch.distributions.Uniform(0.2, 1.0)

    adj = torch.load(graphfp)
    shape = adj.shape
    adj = adj_src_connect(adj,src_nodes)

    rand_mats = torch.tril(uniform_gen.sample(shape))
    weights = (rand_mats + rand_mats.transpose(1,2)) * adj

    steps = bellmanford_alg(adj, weights, src_nodes)
    bellmanford_store(steps, graphfp[:-3]+'_bf.pkl')
    return

## Breadth First Search (Reachability)
def breadthfirstsearch_step(adj, x):
    # assume x is a boolean tensor
    x_next = torch.logical_or(x, torch.logical_and(adj, x.unsqueeze(1).expand_as(adj)).any(2))
    return x_next

def breadthfirstsearch_alg(adj, weights, src, hide_keys):
    steps = []
    adj = (adj != 0)

    # we treat the number of nodes + 1 as infinity
    x_init = torch.zeros((adj.shape[0], adj.shape[1]))
    x_init.scatter_(1, src.unsqueeze(1), 1)

    steps.append((x_init, torch.tensor([]), torch.zeros(adj.shape[0])))
    x_prev = x_init.bool()
    for i in range(adj.shape[1]):
        x = breadthfirstsearch_step(adj, x_prev)
        term = ((~(x^x_prev)).all(1)).long()
        steps.append((x.float(),torch.tensor([]),term))
        x_prev = x
    steps.append((adj, weights, torch.tensor([])))
    return steps

def breadthfirstsearch_store(steps, store_fp):
    steps_numpy = [(t0.numpy(), t1.numpy(), t2.numpy()) for t0, t1, t2 in steps]

    with open(store_fp, 'wb') as f:
        pkl.dump(steps_numpy, f, pkl.HIGHEST_PROTOCOL)

    return


def gen_breadthfirstsearch_data(graphfp, src_nodes):
    uniform_gen = torch.distributions.Uniform(0.2, 1.0)

    adj = torch.load(graphfp)
    shape = adj.shape
    adj = adj_src_connect(adj,src_nodes)

    steps = breadthfirstsearch_alg(adj, src_nodes)
    breadthfirstsearch_store(steps, graphfp[:-3]+'_bfs.pkl')
    return

## Prim's algorithm
def prims_step(adj, weights, x, p):
    # determining which elements are still in the queue & treat non-existing edges as inf
    weights = weights.masked_fill(weights == 0, float('inf'))

    mask1 = (x[:,:,0]==1)
    x_dist = (x[:,:,1]).double()

    # u = Q.pop_min()
    pop_node_idx = torch.argmin(torch.where(~mask1, x_dist, float('inf')), dim=-1).long()
    # update status of popped node
    x_next_status = torch.scatter(x[:,:,0], 1, pop_node_idx.unsqueeze(1), 1)

    # compute the comparison value weight(v,u)
    new_dist = weights
    new_dist = torch.gather(
        new_dist,
        2,
        pop_node_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,adj.shape[1],1)
        ).squeeze().double()
    # mask values already out of the queue
    new_dist = torch.where(~mask1, new_dist, float('inf'))
    x_next_key = torch.min(new_dist, x_dist).squeeze().float()

    # update the predecessor
    p_next = torch.where(x_next_key != x[:,:,1],
                         pop_node_idx.unsqueeze(-1).double(),
                         p)
    return torch.stack([x_next_status, x_next_key], dim=-1), p_next

def prims_alg(adj, weights, src_nodes, hide_keys):
    # initialisation
    x_init = torch.zeros((adj.shape[0], adj.shape[1]))
    # x_init.scatter_(1, src_nodes.unsqueeze(1), 1)
    init_dist = torch.zeros((adj.shape[0], adj.shape[1])).fill_(float('inf'))
    init_dist.scatter_(1, src_nodes.unsqueeze(1), 0)
    p_init = torch.where(init_dist == 0,
                          torch.arange(0,adj.shape[1]).unsqueeze(0).expand_as(x_init).double(),
                          float('inf')
                       )
    x_init = torch.stack([x_init,init_dist],dim=-1)

    steps = []
    steps.append((x_init, p_init, torch.zeros(adj.shape[0])))

    x_prev = x_init
    p_prev = p_init
    term_prev = torch.zeros((adj.shape[0],))
    for i in range(adj.shape[1]):
        x, p = prims_step(adj, weights, x_prev, p_prev)
        term = torch.max(
            1-torch.logical_and(
                torch.logical_and(adj,
                                x[:,:,0].unsqueeze(1).expand_as(adj)
                                ).any(2),
                (1-x[:,:,0])
            ).any(1).long(),
            term_prev
        )
        x = torch.where(term_prev.bool().unsqueeze(1).unsqueeze(2).expand(-1, adj.shape[1],2),
                        x_prev,
                        x.float())
        p = torch.where(term_prev.bool().unsqueeze(1).expand(-1, adj.shape[1]),
                        p_prev,
                        p)
        # hide all the keys of nodes still in the queue
        x_stored = x.clone()
        if hide_keys:
            x_stored[:,:,1] = torch.where(x[:,:,0].bool(),
                                        x[:,:,1],
                                        torch.tensor([float('inf')])
                                        )
        steps.append((x_stored,p, term))
        x_prev = x
        p_prev = p
        term_prev = term
    longest_short_dist = torch.tensor([1]).float()
    for t, _, _ in steps:
        t[:,:,1].masked_scatter_(t[:,:,1] == float('inf'),
                          (longest_short_dist+1).unsqueeze(-1).expand_as(t[:,:,1])[t[:,:,1] == float('inf')])
    steps.append((adj,weights,torch.tensor([])))
    return steps

def prims_store(steps, store_fp):
    steps_numpy = [(t0.numpy(), t1.numpy(), t2.numpy()) for t0, t1, t2 in steps]

    with open(store_fp, 'wb') as f:
        pkl.dump(steps_numpy, f, pkl.HIGHEST_PROTOCOL)

    return

def gen_prims_data(graphfp, src_nodes, weights=None, hide_keys=True):
    uniform_gen = torch.distributions.Uniform(0.2, 1.0)

    adj = torch.load(graphfp)
    shape = adj.shape
    adj = adj_src_connect(adj,src_nodes)

    rand_mats = torch.tril(uniform_gen.sample(shape))
    if weights is None:
        weights = (rand_mats + rand_mats.transpose(1,2)) * adj

    steps = prims_alg(adj, weights, src_nodes, hide_keys)
    prims_store(steps, graphfp[:-3]+'_prims.pkl')
    return

## Dijkstra's algorithm
def dijkstra_step(adj, weights, x, p):
    # determining which elements are still in the queue & treat non-existing edges as inf
    weights = weights.masked_fill(weights == 0, float('inf'))

    mask1 = (x[:,:,0]==1)
    x_dist = (x[:,:,1]).double()

    # u = Q.pop_min()
    pop_node_key, _ = torch.min(torch.where(~mask1, x_dist, float('inf')), dim=-1)
    pop_node_idx = torch.argmin(torch.where(~mask1, x_dist, float('inf')), dim=-1).long().squeeze()
    # update status of popped node
    x_next_status = torch.scatter(x[:,:,0], 1, pop_node_idx.unsqueeze(1), 1)

    # compute the comparison value u.dist + weight
    new_dist = weights + x_dist.unsqueeze(1).repeat(1,adj.shape[1],1)
    new_dist = torch.gather(
        new_dist,
        2,
        pop_node_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,adj.shape[1],1)
        ).squeeze().double()
    # mask values already out of the queue
    new_dist = torch.where(~mask1, new_dist, float('inf'))
    x_next_key = torch.min(new_dist, x_dist).squeeze()

    # update the predecessor
    p_next = torch.where(x_next_key != x[:,:,1],
                         pop_node_idx.unsqueeze(-1).double(),
                         p)
    return torch.stack([x_next_status, x_next_key], dim=-1), p_next

def dijkstra_alg(adj, weights, src_nodes, hide_keys):
    # initialisation
    x_init = torch.zeros((adj.shape[0], adj.shape[1]))
    # x_init.scatter_(1, src_nodes.unsqueeze(1), 1)
    init_dist = torch.zeros((adj.shape[0], adj.shape[1])).fill_(float('inf'))
    init_dist.scatter_(1, src_nodes.unsqueeze(1), 0)
    p_init = torch.where(init_dist == 0,
                          torch.arange(0,adj.shape[1]).unsqueeze(0).expand_as(x_init).double(),
                          float('inf')
                       )
    x_init = torch.stack([x_init,init_dist],dim=-1)
    # torch.zeros((adj.shape[0], adj.shape[1]))
    # start recording
    steps = []
    steps.append((x_init, p_init, torch.zeros(adj.shape[0])))

    x_prev = x_init
    p_prev = p_init
    term_prev = torch.zeros((adj.shape[0]))

    for i in range(adj.shape[1]):
        x, p = dijkstra_step(adj, weights, x_prev, p_prev)
        term = torch.max(
            1-torch.logical_and(
                torch.logical_and(adj,
                                x[:,:,0].unsqueeze(1).expand_as(adj)
                                ).any(2),
                (1-x[:,:,0])
            ).any(1).long(),
            term_prev
        )
        x = torch.where(term_prev.bool().unsqueeze(1).unsqueeze(2).expand(-1, adj.shape[1],2),
                        x_prev,
                        x.float())
        p = torch.where(term_prev.bool().unsqueeze(1).expand(-1, adj.shape[1]),
                        p_prev,
                        p)
        # # hide all the keys of nodes still in the queue
        x_stored = x.clone()
        if hide_keys:
            x_stored[:,:,1] = torch.where(x[:,:,0].bool(),
                                        x[:,:,1],
                                        torch.tensor([float('inf')])
                                        )
        steps.append((x_stored,p,term))
        # steps.append((x,p,term))
        x_prev = x
        p_prev = p
        term_prev = term
    tmp = torch.where(torch.isinf(x[:,:,1]),
                      torch.tensor([0.]).float(),
                      x[:,:,1]
                      )
    longest_short_dist = torch.max(tmp, dim=-1)[0]
    for t, _, _ in steps:
        t[:,:,1].masked_scatter_(
            t[:,:,1] == float('inf'),
            (longest_short_dist+1.).unsqueeze(-1).expand_as(t[:,:,1])[t[:,:,1] == float('inf')]
        )
    steps.append((adj,weights,torch.tensor([])))
    return steps

def dijkstra_store(steps, store_fp):
    steps_numpy = [(t0.numpy(), t1.numpy(), t2.numpy()) for t0, t1, t2 in steps]

    with open(store_fp, 'wb') as f:
        pkl.dump(steps_numpy, f, pkl.HIGHEST_PROTOCOL)

    return

def gen_dijkstra_data(graphfp, src_nodes, weights=None, hide_keys=True):
    uniform_gen = torch.distributions.Uniform(0.2, 1.0)

    adj = torch.load(graphfp)
    shape = adj.shape
    adj = adj_src_connect(adj,src_nodes)

    rand_mats = torch.tril(uniform_gen.sample(shape))
    if weights is None:
        weights = (rand_mats + rand_mats.transpose(1,2)) * adj

    steps = dijkstra_alg(adj, weights, src_nodes, hide_keys)
    dijkstra_store(steps, graphfp[:-3]+'_dijkstra.pkl')
    return

def gen_primsdijkstra_data(graphfp, src_nodes):
    uniform_gen = torch.distributions.Uniform(0.2, 1.0)

    adj = torch.load(graphfp)
    shape = adj.shape
    adj = adj_src_connect(adj,src_nodes)

    rand_mats = torch.tril(uniform_gen.sample(shape))
    weights = (rand_mats + rand_mats.transpose(1,2)) * adj

    # first generate dijkstra
    steps = dijkstra_alg(adj, weights, src_nodes)
    dijkstra_store(steps, graphfp[:-3]+'_dijkstra.pkl')

    # second generate prims
    steps = prims_alg(adj, weights, src_nodes)
    prims_store(steps, graphfp[:-3]+'_prims.pkl')

    return

## Most reliable path algorithm (sequential)
def mostrelseq_step(adj, weights, x, p):
    # determining which elements are still in the queue & treat non-existing edges as inf
    # weights = weights.masked_fill(weights == 0, float('inf'))

    mask1 = (x[:,:,0]==1)
    x_dist = (x[:,:,1]).double()

    # u = Q.pop_min()
    pop_node_key, _ = torch.max(torch.where(~mask1, x_dist, float('-inf')), dim=-1)
    pop_node_idx = torch.argmax(torch.where(~mask1, x_dist, float('-inf')), dim=-1).long().squeeze()
    # update status of popped node
    x_next_status = torch.scatter(x[:,:,0], 1, pop_node_idx.unsqueeze(1), 1)

    # compute the comparison value u.dist + weight
    new_dist = weights * x_dist.unsqueeze(1).repeat(1,adj.shape[1],1)
    new_dist = torch.gather(
        new_dist,
        2,
        pop_node_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,adj.shape[1],1)
        ).squeeze().double()
    # mask values already out of the queue
    new_dist = torch.where(~mask1, new_dist, float('-inf'))
    x_next_key = torch.max(new_dist, x_dist).squeeze()

    # update the predecessor
    p_next = torch.where(x_next_key != x[:,:,1],
                         pop_node_idx.unsqueeze(-1).double(),
                         p)
    return torch.stack([x_next_status, x_next_key], dim=-1), p_next

def mostrelseq_alg(adj, weights, src_nodes, hide_keys):
    # initialisation
    x_init = torch.zeros((adj.shape[0], adj.shape[1]))
    # x_init.scatter_(1, src_nodes.unsqueeze(1), 1)
    init_dist = torch.zeros((adj.shape[0], adj.shape[1])).fill_(0)
    init_dist.scatter_(1, src_nodes.unsqueeze(1), 1)
    p_init = torch.where(init_dist == 1,
                          torch.arange(0,adj.shape[1]).unsqueeze(0).expand_as(x_init).double(),
                          float('inf')
                       )
    x_init = torch.stack([x_init,init_dist],dim=-1)
    # torch.zeros((adj.shape[0], adj.shape[1]))
    # start recording
    steps = []
    steps.append((x_init, p_init, torch.zeros(adj.shape[0])))

    x_prev = x_init
    p_prev = p_init
    term_prev = torch.zeros((adj.shape[0]))

    for i in range(adj.shape[1]):
        x, p = mostrelseq_step(adj, weights, x_prev, p_prev)
        term = torch.max(
            1-torch.logical_and(
                torch.logical_and(adj,
                                x[:,:,0].unsqueeze(1).expand_as(adj)
                                ).any(2),
                (1-x[:,:,0])
            ).any(1).long(),
            term_prev
        )
        x = torch.where(term_prev.bool().unsqueeze(1).unsqueeze(2).expand(-1, adj.shape[1],2),
                        x_prev,
                        x.float())
        p = torch.where(term_prev.bool().unsqueeze(1).expand(-1, adj.shape[1]),
                        p_prev,
                        p)
        # # hide all the keys of nodes still in the queue
        x_stored = x.clone()
        if hide_keys:
            x_stored[:,:,1] = torch.where(x[:,:,0].bool(),
                                          x[:,:,1],
                                        torch.tensor([0.])
                                        )
        steps.append((x_stored,p,term))
        # steps.append((x,p,term))
        x_prev = x
        p_prev = p
        term_prev = term
    # tmp = torch.where(torch.isinf(x[:,:,1]),
    #                   torch.tensor([0.]).float(),
    #                   x[:,:,1]
    #                   )
    # longest_short_dist = torch.max(tmp, dim=-1)[0]
    # for t, _, _ in steps:
    #     t[:,:,1].masked_scatter_(
    #         t[:,:,1] == float('inf'),
    #         (longest_short_dist+1.).unsqueeze(-1).expand_as(t[:,:,1])[t[:,:,1] == float('inf')]
    #     )
    steps.append((adj,weights,torch.tensor([])))
    return steps

def mostrelseq_store(steps, store_fp):
    steps_numpy = [(t0.numpy(), t1.numpy(), t2.numpy()) for t0, t1, t2 in steps]

    with open(store_fp, 'wb') as f:
        pkl.dump(steps_numpy, f, pkl.HIGHEST_PROTOCOL)

    return

def gen_mostrelseq_data(graphfp, src_nodes, weights=None, hide_keys=True):
    uniform_gen = torch.distributions.Uniform(0.2, 1.0)

    adj = torch.load(graphfp)
    shape = adj.shape
    adj = adj_src_connect(adj,src_nodes)

    rand_mats = torch.tril(uniform_gen.sample(shape))
    if weights is None:
        weights = (rand_mats + rand_mats.transpose(1,2)) * adj

    steps = mostrelseq_alg(adj, weights, src_nodes, hide_keys)
    mostrelseq_store(steps, graphfp[:-3]+'_dijkstra.pkl')
    return

## Most reliable path algorithm (par)
def mostrelpar_step(adj, weights, x, p):
    x_neigh = x.unsqueeze(1).expand_as(adj).transpose(1,2).masked_fill(~adj, 0)
    # weights = weights.masked_fill(weights == 0, adj.shape[1]+1)
    x_next = torch.max(x, torch.max(x_neigh * weights, dim=1)[0])
    p_next = torch.where((x_next == 0) + (x_next == 1),
                       # torch.arange(0,adj.shape[1]).unsqueeze(0).expand_as(x_next).long(),
                       p,
                       torch.argmax(x_neigh * weights, dim=1).float()
                         ).float()
    return x_next, p_next

def mostrelpar_alg(adj, weights, src, hide_keys):
    steps = []
    adj = (adj != 0)

    # we treat the number of nodes + 1 as infinity
    x_init = torch.zeros((adj.shape[0], adj.shape[1])).fill_(0)
    x_init.scatter_(1, src.unsqueeze(1), 1)
    p_init = torch.where(x_init == 1,
                          torch.arange(0,adj.shape[1]).unsqueeze(0).expand_as(x_init).double(),
                          float('inf')
                       ).float()

    steps.append((x_init, torch.tensor([]), torch.zeros(adj.shape[0])))

    x_prev = x_init
    p_prev = p_init
    for i in range(adj.shape[1]):
        x, p = mostrelpar_step(adj, weights, x_prev, p_prev)
        term = (((x-x_prev) == 0).all(1)).long()
        steps.append((x,p, term))
        x_prev = x
        p_prev = p
    steps.append((adj,weights,torch.tensor([])))
    return steps

def mostrelpar_store(steps, store_fp):
    steps_numpy = [(t0.numpy(), t1.numpy(), t2.numpy()) for t0, t1, t2 in steps]

    with open(store_fp, 'wb') as f:
        pkl.dump(steps_numpy, f, pkl.HIGHEST_PROTOCOL)

    return

def gen_mostrelpar_data(graphfp, src_nodes):
    uniform_gen = torch.distributions.Uniform(0.2, 1.0)

    adj = torch.load(graphfp)
    shape = adj.shape
    adj = adj_src_connect(adj,src_nodes)

    rand_mats = torch.tril(uniform_gen.sample(shape))
    weights = (rand_mats + rand_mats.transpose(1,2)) * adj

    steps = mostrelpar_alg(adj, weights, src_nodes)
    mostrelpar_store(steps, graphfp[:-3]+'_mostrelpar.pkl')
    return

## Depth-first search (Reachability)
def dfs_step(adj, x, p):
    # determining which elements are still in queue
    x_queue = torch.where(x[:,:,0]==0,
                          x[:,:,1],
                          1-x[:,:,0]) # just want to write 0
    # u = Q.pop_min()
    pop_node = torch.argmax(x_queue,dim=1)
    cur_dist, _ = torch.max(x_queue, dim=1)
    # update status of popped node
    x_next_status = torch.scatter(x[:,:,0], 1, pop_node.unsqueeze(1), 1)
    # gather the neighbours
    pop_node_neigh = torch.gather(
        adj,
        2,
        pop_node.unsqueeze(-1).unsqueeze(-1).repeat(1,adj.shape[1],1)
        ).squeeze()*(cur_dist+1).unsqueeze(1)*(1-x[:,:,0])
    # update the neighbours, should only change key once
    x_next_key = torch.where(torch.logical_and((x_queue == 0),pop_node_neigh>0),
                             pop_node_neigh,
                             x[:,:,1])
    # update the predecessor
    p_next = torch.where(x_next_key != x[:,:,1],
                         pop_node.unsqueeze(-1).float(),
                         p)
    return torch.stack([x_next_status, x_next_key],dim=-1), p_next

def dfs_alg(adj, weights, src, hide_keys):
    steps = []
    adj = (adj != 0)

    # node key 0 is unreachable
    x_init_0 = torch.zeros((adj.shape[0], adj.shape[1]))
    x_init_1 = torch.zeros((adj.shape[0], adj.shape[1]))
    x_init_1.scatter_(1, src.unsqueeze(1), 1)
    # x_init_1.scatter_(1, src.unsqueeze(1), 1)
    x_init = torch.stack([x_init_0, x_init_1], dim=-1)

    p_init = torch.where(x_init_1 == 1,
                          torch.arange(0,adj.shape[1]).unsqueeze(0).expand_as(x_init_1).double(),
                          float('inf')
                       ).float()

    steps.append((x_init, p_init, torch.zeros(adj.shape[0])))
    x_prev = x_init
    p_prev = p_init
    term_prev = torch.zeros((adj.shape[0],))
    for i in range(adj.shape[1]):
        x, p = dfs_step(adj, x_prev, p_prev)
        term = torch.max(
            1-torch.logical_and(
                torch.logical_and(adj,
                                x[:,:,0].unsqueeze(1).expand_as(adj)
                                ).any(2),
                (1-x[:,:,0])
            ).any(1).long(),
            term_prev
        )
        x = torch.where(term_prev.bool().unsqueeze(1).unsqueeze(2).expand(-1, adj.shape[1],2),
                        x_prev,
                        x.float())
        p = torch.where(term_prev.bool().unsqueeze(1).expand(-1, adj.shape[1]),
                        p_prev,
                        p)
        # # hide all the keys of nodes still in the queue
        x_stored = x.clone()
        if hide_keys:
            x_stored[:,:,1] = torch.where(x[:,:,0].bool(),
                                          x[:,:,1],
                                        torch.tensor([0.])
                                        )
        steps.append((x_stored,p,term))
        x_prev = x
        p_prev = p
        term_prev = term
    steps.append((adj, weights, torch.tensor([])))
    return steps

def dfs_store(steps, store_fp):
    steps_numpy = [(t0.numpy(), t1.numpy(), t2.numpy()) for t0, t1, t2 in steps]

    with open(store_fp, 'wb') as f:
        pkl.dump(steps_numpy, f, pkl.HIGHEST_PROTOCOL)

    return

def gen_dfs_data(graphfp, src_nodes):
    uniform_gen = torch.distributions.Uniform(0.2, 1.0)

    adj = torch.load(graphfp)
    shape = adj.shape

    steps = dfs_alg(adj, None, src_nodes)
    dfs_store(steps, graphfp[:-3]+'_dfs.pkl')
    return

## Widest path algorithm
def widest_step(adj, weights, x, p):
    # determining which elements are still in the queue & treat non-existing edges as inf
    # weights = weights.masked_fill(weights == 0, float('inf'))

    mask1 = (x[:,:,0]==1)
    x_dist = (x[:,:,1]).double()

    # u = Q.pop_min()
    pop_node_key, _ = torch.max(torch.where(~mask1, x_dist, -float('inf')), dim=-1)
    pop_node_idx = torch.argmax(torch.where(~mask1, x_dist, -float('inf')), dim=-1).long().squeeze()
    # update status of popped node
    x_next_status = torch.scatter(x[:,:,0], 1, pop_node_idx.unsqueeze(1), 1)

    #
    new_dist = torch.min(weights, x_dist.unsqueeze(1).repeat(1,adj.shape[1],1))    # adeac prev + instead of min
    new_dist = torch.gather(
        new_dist,
        2,
        pop_node_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,adj.shape[1],1)
        ).squeeze().double()

    # mask values already out of the queue
    new_dist = torch.where(~mask1, new_dist, -float('inf'))
    x_next_key = torch.max(new_dist, x_dist).squeeze()   #adeac prev min

    # update the predecessor
    p_next = torch.where(x_next_key != x[:,:,1],
                         pop_node_idx.unsqueeze(-1).double(),
                         p)
    return torch.stack([x_next_status, x_next_key], dim=-1), p_next

def widest_alg(adj, weights, src_nodes, hide_keys):
    # initialisation
    x_init = torch.zeros((adj.shape[0], adj.shape[1]))
    # x_init.scatter_(1, src_nodes.unsqueeze(1), 1)
    init_dist = torch.zeros((adj.shape[0], adj.shape[1]))   # adeac: prev .fill_(float('inf'))
    init_dist.scatter_(1, src_nodes.unsqueeze(1), float('inf'))        # adeac: prev (src=)0
    p_init = torch.where(init_dist == float('inf'),  # adeac: prev == 0^M
                         torch.arange(0, adj.shape[1]).unsqueeze(0).expand_as(x_init).double(),
                         float('inf')
                       )

    x_init = torch.stack([x_init,init_dist],dim=-1)
    # torch.zeros((adj.shape[0], adj.shape[1]))
    # start recording
    steps = []
    steps.append((x_init, p_init, torch.zeros(adj.shape[0])))

    x_prev = x_init
    p_prev = p_init
    term_prev = torch.zeros((adj.shape[0]))

    for i in range(adj.shape[1]):
        x, p = widest_step(adj, weights, x_prev, p_prev)
        term = torch.max(
            1-torch.logical_and(
                torch.logical_and(adj,
                                x[:,:,0].unsqueeze(1).expand_as(adj)
                                ).any(2),
                (1-x[:,:,0])
            ).any(1).long(),
            term_prev
        )
        x = torch.where(term_prev.bool().unsqueeze(1).unsqueeze(2).expand(-1, adj.shape[1],2),
                        x_prev,
                        x.float())
        p = torch.where(term_prev.bool().unsqueeze(1).expand(-1, adj.shape[1]),
                        p_prev,
                        p)
        x_stored = x.clone()
        # make src.width = 0 (as x[src, 0] = 1 so assumed it would impact loss).
        assert np.all(np.asarray(x_stored[np.arange(src_nodes.shape[0]), src_nodes, 1]) == float('inf'))
        x_stored[np.arange(src_nodes.shape[0]), src_nodes, 1] = 0.
        # # hide all the keys of nodes still in the queue
        x_stored[:,:,1] = torch.where(x[:,:,0].bool(),
                                      x[:,:,1],
                                      torch.tensor([float('inf')])
                                      )
        steps.append((x_stored,p,term))
        # steps.append((x,p,term))
        x_prev = x
        p_prev = p
        term_prev = term
    tmp = torch.where(torch.isinf(x[:,:,1]),
                      torch.tensor([0.]).float(),
                      x[:,:,1]
                      )
    longest_short_dist = torch.max(tmp, dim=-1)[0]
    for t, _, _ in steps:
        t[:,:,1].masked_scatter_(
            t[:,:,1] == float('inf'),
            (longest_short_dist+1.).unsqueeze(-1).expand_as(t[:,:,1])[t[:,:,1] == float('inf')]
        )
    steps.append((adj,weights,torch.tensor([])))
    return steps

def widest_store(steps, store_fp):
    steps_numpy = [(t0.numpy(), t1.numpy(), t2.numpy()) for t0, t1, t2 in steps]

    with open(store_fp, 'wb') as f:
        pkl.dump(steps_numpy, f, pkl.HIGHEST_PROTOCOL)

    return

def gen_widest_data(graphfp, src_nodes, weights=None, hide_keys=True):
    uniform_gen = torch.distributions.Uniform(0.2, 1.0)

    adj = torch.load(graphfp)
    shape = adj.shape
    adj = adj_src_connect(adj,src_nodes)

    rand_mats = torch.tril(uniform_gen.sample(shape))
    if weights is None:
        weights = (rand_mats + rand_mats.transpose(1,2)) * adj

    steps = widest_alg(adj, weights, src_nodes, hide_keys)
    widest_store(steps, graphfp[:-3]+'_widest.pkl')
    return

## Widest path algorithm (par)
def widestpar_step(adj, weights, x, p):
    x_neigh = x.unsqueeze(1).expand_as(adj).transpose(1,2).masked_fill(~adj, 0)
    # weights = weights.masked_fill(weights == 0, adj.shape[1]+1)
    x_next = torch.max(x, torch.max(torch.min(x_neigh,weights), dim=1)[0])
    p_next = torch.where((x_next == 0) + (x_next == 1),
                       # torch.arange(0,adj.shape[1]).unsqueeze(0).expand_as(x_next).long(),
                       p,
                       torch.argmax(torch.min(x_neigh,weights), dim=1).float()
                         ).float()
    return x_next, p_next

def widestpar_alg(adj, weights, src, hide_keys):
    steps = []
    adj = (adj != 0)

    # we treat the number of nodes + 1 as infinity
    x_init = torch.zeros((adj.shape[0], adj.shape[1])).fill_(0)
    x_init.scatter_(1, src.unsqueeze(1), 1)
    p_init = torch.where(x_init == 1,
                          torch.arange(0,adj.shape[1]).unsqueeze(0).expand_as(x_init).double(),
                          float('inf')
                       ).float()

    steps.append((x_init, torch.tensor([]), torch.zeros(adj.shape[0])))

    x_prev = x_init
    p_prev = p_init
    for i in range(adj.shape[1]):
        x, p = widestpar_step(adj, weights, x_prev, p_prev)
        term = (((x-x_prev) == 0).all(1)).long()
        steps.append((x,p, term))
        x_prev = x
        p_prev = p
    steps.append((adj,weights,torch.tensor([])))
    return steps

def widestpar_store(steps, store_fp):
    steps_numpy = [(t0.numpy(), t1.numpy(), t2.numpy()) for t0, t1, t2 in steps]

    with open(store_fp, 'wb') as f:
        pkl.dump(steps_numpy, f, pkl.HIGHEST_PROTOCOL)

    return

def gen_widestpar_data(graphfp, src_nodes):
    uniform_gen = torch.distributions.Uniform(0.2, 1.0)

    adj = torch.load(graphfp)
    shape = adj.shape
    adj = adj_src_connect(adj,src_nodes)

    rand_mats = torch.tril(uniform_gen.sample(shape))
    weights = (rand_mats + rand_mats.transpose(1,2)) * adj

    steps = widestpar_alg(adj, weights, src_nodes)
    widestpar_store(steps, graphfp[:-3]+'_widestpar.pkl')
    return

### UTILS (if grows too much give it's own file)
def adj_src_connect(adj, src_nodes):

    adj_connect = torch.zeros(adj.shape[:-1])
    idx = torch.arange(adj.shape[1]-1,-1,-1).repeat(adj.shape[0]//adj.shape[1]+1)[:adj.shape[0]]
    pdb.set_trace()
    # second index set incase src nodes over laps with the other one
    idx2 = torch.fmod(
        torch.arange(adj.shape[1],0,-1).repeat(adj.shape[0]//adj.shape[1]+1)[:adj.shape[0]],
        adj.shape[1]
    )
    # now ensure that src_nodes are not the same as index
    idx = torch.where(idx == src_nodes,
                      idx2,
                      idx
                      ).unsqueeze(-1)
    adj_implant = torch.scatter(adj_connect, 1, idx, 1)
    adj_to_mod = (torch.gather(adj.long(),
                               1,
                               src_nodes.unsqueeze(-1).unsqueeze(-1).repeat(1,1,adj.shape[1])
                               ).squeeze() == 0).all(1)
    adj_connect = torch.scatter(adj.long(),
                                1,
                                src_nodes.unsqueeze(-1).unsqueeze(-1).repeat(1,1,adj.shape[1]),
                                adj_implant.long().unsqueeze(1)
                                )
    adj_connect.scatter_(
                   2,
                   src_nodes.unsqueeze(-1).unsqueeze(-1).repeat(1,adj.shape[1],1),
                   adj_implant.long().unsqueeze(-1)
                   )
    adj = torch.where(adj_to_mod.unsqueeze(-1).unsqueeze(-1).expand_as(adj),
                      adj_connect,
                      adj.long()
                      ).bool()
    return adj
