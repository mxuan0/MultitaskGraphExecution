import math
import operator
from functools import reduce
from sympy.ntheory import factorint

import torch

def gen_erdos_renyi(uniform_gen, n_graphs, n_nodes, save_fp=None, directed=False):
    """
    uniform_gen: a torch uniform distribution [0.0, 1.0] we can sample from
    n_graphs: int
    n_nodes: int
    save_fp: string to file location

    returns erdos renyi random graphs
    """
    edge_prob = min(math.log(float(n_nodes), 2)/float(n_nodes), 0.5) # uniform_gen.sample((n_graphs,)) * 0.9 + 0.1 # we don't want empty graphs
    if directed:
        cutoff = edge_prob
    else:
        cutoff = math.sqrt(edge_prob) # torch.sqrt(edge_prob)
    # cutoff = cutoff.unsqueeze(1).unsqueeze(2).repeat(1, n_nodes, n_nodes)
    rand_adj = uniform_gen.sample((n_graphs, n_nodes, n_nodes))
    rand_cutoff_adj = rand_adj < cutoff
    if directed:
        rand_sym_adj = rand_cutoff_adj
    else:
        rand_sym_adj = torch.logical_and(rand_cutoff_adj, rand_cutoff_adj.transpose(1,2))
    # ensuring there are no self-loops
    rand_sym_adj[:, torch.arange(n_nodes), torch.arange(n_nodes)] = 0

    # saving the tensor
    if save_fp is not None:
        torch.save(rand_sym_adj, save_fp+"erdosrenyi" + str(n_graphs) + "_" + str(n_nodes) + ".pt")

    return rand_sym_adj.int()

def gen_barabasi_albert(uniform_gen, n_graphs, n_nodes, m_attach, save_fp=None):
    rand_src = uniform_gen.sample((n_graphs, n_nodes-m_attach,m_attach))
    node_degs = torch.zeros((n_graphs, n_nodes))
    adj = torch.zeros((n_graphs, n_nodes, n_nodes))

    ## adding the first two nodes to simplify loop
    node_degs[:,(n_nodes-m_attach):] = 1
    node_degs[:,(n_nodes-m_attach):] = 1

    ## index into random vector
    idx = 0

    for i in range(n_nodes-m_attach):
        probs = node_degs.clone()
        edges = torch.zeros((n_graphs, n_nodes))
        for j in range(m_attach):
            probs_cdf = probs.cumsum(dim=1)
            probs_cdf = probs_cdf/probs_cdf[:,-1].unsqueeze(1)
            idx = n_nodes - (probs_cdf > rand_src[:,i,j].unsqueeze(1)).sum(dim=1)
            edges.scatter_(1, idx.unsqueeze(1), 1)
            probs.scatter_(1, idx.unsqueeze(1), 0)
        node_degs = node_degs + edges
        node_degs[:,i] = m_attach
        adj[:,i,:] = adj[:,i,:] + edges
        adj[:,:,i] = adj[:,:,i] + edges

    # saving the tensor
    if save_fp is not None:
        torch.save(adj, save_fp+"barabasialbert" + str(n_graphs) + "_" + str(n_nodes) + ".pt")

    return adj

def powerset(seq):
    return [x for ps in powerset(seq[1:]) for x in ([seq[0]] + ps, ps)] if seq else [[]]

def prod(seq):
    return reduce(operator.mul, seq, 1)

def sorted_2tuple(a, b):
    if a > b:
        return (b, a)
    else:
        return (a, b)

def four_neighbours(x,y,length,width):
    # -1 means doesn't exist
    base = [-1, -1, -1, -1]
    if y != 0:
        base[0] = width*x + y - 1
    if y < width-1:
        base[1] = width*x + y + 1
    if x != 0:
        base[2] = width*(x-1) + y
    if x < length-1:
        base[3] = width*(x+1) + y
    return base

def gen_twod_grid(uniform_gen, n_graphs, n_nodes, save_fp=None, directed=False):
    factors = factorint(n_nodes)
    factors = [ k for k, v in factors.items() for _ in range(v) ]
    factors = powerset(factors)
    shapes = list(set([ sorted_2tuple(prod(seq), n_nodes // prod(seq)) for seq in factors ]))
    if len(shapes) > 1:
        shapes = list(filter(lambda t: t[0] != 1, shapes))

    grid_shape = torch.argmax(uniform_gen.sample((n_graphs, len(shapes))), dim=-1)

    # generate position of each node
    xy = [ [ (i,j,*shapes[idx]) for i in range(shapes[idx][0]) for j in range(shapes[idx][1]) ] for idx in grid_shape.tolist() ]
    neighbours = [ [ four_neighbours(*t) for t in g] for g in xy]
    values = [[[ 0 if i < 0 else 1 for i in node] for node in g] for g in neighbours ]
    neighbours = [ [ [  i if n == -1 else n for n in g[i] ] for i in range(len(g))] for g in neighbours ]

    # generate neighbour indices
    values = torch.tensor(values)
    neighbours = torch.tensor(neighbours)

    # generate grid if generated sizes
    adj = torch.zeros((n_graphs, n_nodes, n_nodes))
    adj = adj.long().scatter_(2, neighbours, values)

    # permuting the nodes, doesn't change graph only representation of said graph
    perm_matrix = uniform_gen.sample((n_graphs, n_nodes, n_nodes))
    for i in range(n_nodes):
        idx = torch.argmax(perm_matrix[:,i,:], dim=-1)
        perm_matrix[:,i,:] = 0
        perm_matrix.scatter_(2, idx.unsqueeze(-1).unsqueeze(-1).repeat(1, n_nodes, 1), 0)
        perm_matrix[:,i,:].scatter_(1, idx.unsqueeze(-1), 1)

    perm_matrix = perm_matrix.long()
    adj = torch.bmm(perm_matrix, adj)
    adj = torch.bmm(adj, perm_matrix.transpose(1,2)).bool()

    if save_fp is not None:
        torch.save(adj, save_fp+"twodgrid" + str(n_graphs) + "_" + str(n_nodes) + ".pt")

    return adj
