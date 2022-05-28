import pickle as pkl

import torch
from torch.utils.data import Dataset, IterableDataset
import pdb
class MultiAlgo(Dataset):
    """
    Data produced by algorithms such as BFS solving the Reachability task on
    various graphs including each step of the algorithm as well as solving the
    shortest path problem simultaeneously.

    Important!! All graphs must be of the size in a given dataset!
    """

    def __init__(self, logger, datafp, algos, name='Train'):
        """
        logger: for printing out information about the dataset
        datafp: list of one or more strings representing filepaths to data to be loaded
        """
        self.log = logger
        self.name = name
        self.algos = algos
        self.n_files = len(datafp)
        if self.n_files < 1:
            logger.error("No filepaths have been passed to the dataset constructor!")

        self.data = []
        self.pred_exists = True
        #pdb.set_trace()
        for fp  in datafp:
            self.data.append(self.load_data(logger, fp))

        self.length = sum([t[0].shape[0] for t in self.data])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        quo, rem = divmod(idx, self.n_files)
        return self.data[rem][0][quo], self.data[rem][1][quo], self.data[rem][2][quo],\
             self.data[rem][3][quo] if self.pred_exists else torch.tensor([0]),\
             self.data[rem][4][quo]

    def load_data(self, logger, fp):
        state_list = []
        pred_list = []
        # first algo is done differently to get the graphs
        first_algo = self.algos[0]
        with open(fp+f'_{first_algo}.pkl', 'rb') as f:
            steps_numpy = pkl.load(f)
        #pdb.set_trace()
        steps = [(torch.tensor(t0), torch.tensor(t1), torch.tensor(t2)) for t0, t1, t2 in steps_numpy]
        adj, weights, _ = steps[-1]
        state, pred, term = zip(*(steps[:-1]))
        state_tensor = torch.stack(state, dim=-1)
        # dealing with algos with nodedim that have the shape BATCH x NNODES instead of
        # BATCH x NNODES x NODEDIM
        if state_tensor.ndim < 4:
            state_tensor = state_tensor.unsqueeze(2)
        if pred[1].size() != (0,):
            pred_tensor = torch.stack(pred[1:], dim=-1)
            self.pred_exists = True
        else:
            pred_tensor = None
            self.pred_exists = False
        term_tensor = torch.stack(term, dim=-1)

        state_list.append(state_tensor)
        pred_list.append(pred_tensor)
        min_term_tensor = term_tensor

        for algo in self.algos[1:]:
            with open(fp+f'_{algo}.pkl', 'rb') as f:
                steps_numpy = pkl.load(f)

            steps = [(torch.tensor(t0), torch.tensor(t1), torch.tensor(t2)) for t0, t1, t2 in steps_numpy]
            next_adj, next_weights, _ = steps[-1]
            if (not torch.equal(next_adj, adj)) and (not torch.equal(next_weights, weights)):
                self.log.exception("The graphs are different. Erroneous file: " + fp)
            state, pred, term = zip(*(steps[:-1]))
            state_tensor = torch.stack(state, dim=-1)
            if state_tensor.ndim < 4:
                state_tensor = state_tensor.unsqueeze(2)
            if pred[1].size() != (0,):
                pred_tensor = torch.stack(pred[1:], dim=-1)
            else:
                pred_tensor = None
            term_tensor = torch.stack(term, dim=-1)

            state_list.append(state_tensor)
            pred_list.append(pred_tensor)
            # ensure our termination criterion is the longest one
            min_term_tensor = torch.min(min_term_tensor, term_tensor)

        # concatening the stack and pred tensor
        state_total = torch.cat(state_list, dim=-2)
        pred_list = [pred for pred in pred_list if pred is not None]
        if len(pred_list) > 0:
            pred_total = torch.stack(pred_list, dim=-2)
            self.pred_exists = True
        else:
            pred_total = None
            self.pred_exists = False

        return adj, weights, state_total, pred_total, min_term_tensor

def collate_multi_algo(batch):
    adj = torch.stack([item[0] for item in batch], dim=0)
    weights = torch.stack([item[1] for item in batch], dim=0)
    state = torch.stack([item[2] for item in batch], dim=0)
    pred = torch.stack([item[3] for item in batch], dim=0)
    term = torch.stack([item[4] for item in batch], dim=0)
    return adj, weights, state, pred, term

class BFSteps(Dataset):
    """
    Data produced by algorithms such as BFS solving the Reachability task on
    various graphs including each step of the algorithm as well as solving the
    shortest path problem simultaeneously.
    Important!! All graphs must be of the size in a given dataset!
    """

    def __init__(self, logger, datafp, name='Train'):
        """
        logger: for printing out information about the dataset
        datafp: list of one or more strings representing filepaths to data to be loaded
        """
        self.log = logger
        self.name = name
        self.n_files = len(datafp)
        if self.n_files < 1:
            logger.error("No filepaths have been passed to the ShortestPathSteps dataset constructor!")

        self.data = []

        for fp  in datafp:
            self.data.append(self.load_data(logger, fp))

        self.length = sum([t[0].shape[0] for t in self.data])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        quo, rem = divmod(idx, self.n_files)
        return self.data[rem][0][quo], self.data[rem][1][quo], \
            self.data[rem][2][quo], self.data[rem][3][quo], \
            self.data[rem][4][quo]

    def load_data(self, logger, fp):
        # loading bellman-ford
        with open(fp+'_bf.pkl', 'rb') as f:
            bf_steps_numpy = pkl.load(f)

        bf_steps = [(torch.tensor(t0), torch.tensor(t1), torch.tensor(t2)) for t0, t1, t2 in bf_steps_numpy]
        adj, weights,_ = bf_steps[-1]
        bf_state, pred, bf_term = zip(*(bf_steps[:-1]))
        bf_state_tensor = torch.stack(bf_state, dim=-1)
        if bf_state_tensor.ndim < 4:
            bf_state_tensor = bf_state_tensor.unsqueeze(2)
        pred_tensor = torch.stack(pred[1:], dim=-1).unsqueeze(-2)
        bf_term_tensor = torch.stack(bf_term, dim=-1)

        return adj, weights, bf_state_tensor, pred_tensor, bf_term_tensor

def collate_bf(batch):
    adj = torch.stack([item[0] for item in batch], dim=0)
    weights = torch.stack([item[1] for item in batch], dim=0)
    bf_state = torch.stack([item[2] for item in batch], dim=0)
    pred = torch.stack([item[3] for item in batch], dim=0)
    term = torch.stack([item[4] for item in batch], dim=0)
    return adj, weights,  bf_state, pred, term

class ReachabilitySteps(Dataset):
    """
    Data produced by algorithms such as BFS solving the Reachability task on
    various graphs including each step of the algorithm as well as solving the
    shortest path problem simultaeneously.
    Important!! All graphs must be of the size in a given dataset!
    """

    def __init__(self, logger, datafp, name='Train'):
        """
        logger: for printing out information about the dataset
        datafp: list of one or more strings representing filepaths to data to be loaded
        """
        self.log = logger
        self.name = name
        self.n_files = len(datafp)
        if self.n_files < 1:
            logger.error("No filepaths have been passed to the ShortestPathSteps dataset constructor!")

        self.data = []

        for fp  in datafp:
            self.data.append(self.load_data(logger, fp))

        self.length = sum([t[0].shape[0] for t in self.data])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        quo, rem = divmod(idx, self.n_files)
        return self.data[rem][0][quo], self.data[rem][1][quo], self.data[rem][2][quo], \
            torch.tensor([0]), self.data[rem][3][quo]

    def load_data(self, logger, fp):
        # loading breadth first search
        with open(fp+'_bfs.pkl', 'rb') as f:
            bfs_steps_numpy = pkl.load(f)

        bfs_steps = [(torch.tensor(t0), torch.tensor(t1), torch.tensor(t2)) for t0, t1, t2 in bfs_steps_numpy]
        adj, weights, _ = bfs_steps[-1]
        bfs_state, _, bfs_term = zip(*(bfs_steps[:-1]))
        bfs_state_tensor = torch.stack(bfs_state, dim=-1)

        term_tensor = torch.stack(bfs_term, dim=-1)

        return adj, weights, bfs_state_tensor.unsqueeze(2), term_tensor

def collate_reach(batch):
    adj = torch.stack([item[0] for item in batch], dim=0)
    weights = torch.stack([item[1] for item in batch], dim=0)
    bfs_state = torch.stack([item[2] for item in batch], dim=0)
    term = torch.stack([item[4] for item in batch], dim=0)
    return adj, weights, bfs_state, torch.tensor([0]), term
    
class BFSBFSteps(Dataset):
    """
    Data produced by algorithms such as BFS solving the Reachability task on
    various graphs including each step of the algorithm as well as solving the
    shortest path problem simultaeneously.

    Important!! All graphs must be of the size in a given dataset!
    """

    def __init__(self, logger, datafp, name='Train'):
        """
        logger: for printing out information about the dataset
        datafp: list of one or more strings representing filepaths to data to be loaded
        """
        self.log = logger
        self.name = name
        self.n_files = len(datafp)
        if self.n_files < 1:
            logger.error("No filepaths have been passed to the ShortestPathSteps dataset constructor!")

        self.data = []

        for fp  in datafp:
            self.data.append(self.load_data(logger, fp))

        self.length = sum([t[0].shape[0] for t in self.data])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        quo, rem = divmod(idx, self.n_files)
        return self.data[rem][0][quo], self.data[rem][1][quo], \
            self.data[rem][2][quo], self.data[rem][3][quo], \
            self.data[rem][4][quo], self.data[rem][5][quo]

    def load_data(self, logger, fp):
        # loading bellman-ford
        with open(fp+'_bf.pkl', 'rb') as f:
            bf_steps_numpy = pkl.load(f)

        bf_steps = [(torch.tensor(t0), torch.tensor(t1), torch.tensor(t2)) for t0, t1, t2 in bf_steps_numpy]
        adj, weights,_ = bf_steps[-1]
        bf_state, pred, bf_term = zip(*(bf_steps[:-1]))
        bf_state_tensor = torch.stack(bf_state, dim=-1)
        pred_tensor = torch.stack(pred[1:], dim=-1)
        bf_term_tensor = torch.stack(bf_term, dim=-1)

        # loading breadth first search
        with open(fp+'_bfs.pkl', 'rb') as f:
            bfs_steps_numpy = pkl.load(f)

        bfs_steps = [(torch.tensor(t0), torch.tensor(t1)) for t0, t1 in bfs_steps_numpy]
        adj_bfs, _ = bfs_steps[-1]
        if not torch.equal(adj_bfs, adj):
            self.log.exception("The graphs for bellmanford and bfs are different. Erroneous file: " + fp)
        bfs_state, bfs_term = zip(*(bfs_steps[:-1]))
        bfs_state_tensor = torch.stack(bfs_state, dim=-1)
        bfs_term_tensor = torch.stack(bfs_term, dim=-1)
        # we are done when the slower algorithm is done
        term_tensor = torch.min(bfs_term_tensor, bf_term_tensor)

        return adj, weights, bfs_state_tensor, bf_state_tensor, pred_tensor, term_tensor

def collate_bfsbf(batch):
    adj = torch.stack([item[0] for item in batch], dim=0)
    weights = torch.stack([item[1] for item in batch], dim=0)
    bfs_state = torch.stack([item[2] for item in batch], dim=0)
    bf_state = torch.stack([item[3] for item in batch], dim=0)
    pred = torch.stack([item[4] for item in batch], dim=0)
    term = torch.stack([item[5] for item in batch], dim=0)
    return adj, weights, bfs_state, bf_state, pred, term

class PrimsSteps(Dataset):
    """
    Data produced by Prim's algorithm, miminum spanning tree task task on
    various graphs including each step of the algorithm as well as solving the
    shortest path problem simultaeneously.

    Important!! All graphs must be of the size in a given dataset!
    """

    def __init__(self, logger, datafp, name='Train'):
        """
        logger: for printing out information about the dataset
        datafp: list of one or more strings representing filepaths to data to be loaded
        """
        self.log = logger
        self.name = name
        self.n_files = len(datafp)
        if self.n_files < 1:
            logger.error("No filepaths have been passed to the ShortestPathSteps dataset constructor!")

        self.data = []

        for fp  in datafp:
            self.data.append(self.load_data(logger, fp))

        self.length = sum([t[0].shape[0] for t in self.data])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        quo, rem = divmod(idx, self.n_files)
        return self.data[rem][0][quo], self.data[rem][1][quo], \
            self.data[rem][2][quo], self.data[rem][3][quo], \
            self.data[rem][4][quo]

    def load_data(self, logger, fp):
        with open(fp+'_prims.pkl', 'rb') as f:
            prims_steps_numpy = pkl.load(f)

        prims_steps = [(torch.tensor(t0), torch.tensor(t1), torch.tensor(t2)) for t0, t1, t2 in prims_steps_numpy]
        adj, weights,_ = prims_steps[-1]
        prims_state, pred, prims_term = zip(*(prims_steps[:-1]))
        prims_state_tensor = torch.stack(prims_state, dim=-1)
        pred_tensor = torch.stack(pred, dim=-1)
        term_tensor = torch.stack(prims_term, dim=-1)

        return adj, weights, prims_state_tensor, pred_tensor, term_tensor


def collate_prims(batch):
    adj = torch.stack([item[0] for item in batch], dim=0)
    weights = torch.stack([item[1] for item in batch], dim=0)
    prims_state = torch.stack([item[2] for item in batch], dim=0)
    pred = torch.stack([item[3] for item in batch], dim=0)
    term = torch.stack([item[4] for item in batch], dim=0)
    return adj, weights, prims_state, pred, term

class DijkstraSteps(Dataset):
    """
    Data produced by Prim's algorithm, miminum spanning tree task task on
    various graphs including each step of the algorithm as well as solving the
    shortest path problem simultaeneously.

    Important!! All graphs must be of the size in a given dataset!
    """

    def __init__(self, logger, datafp, name='Train'):
        """
        logger: for printing out information about the dataset
        datafp: list of one or more strings representing filepaths to data to be loaded
        """
        self.log = logger
        self.name = name
        self.n_files = len(datafp)
        if self.n_files < 1:
            logger.error("No filepaths have been passed to the ShortestPathSteps dataset constructor!")

        self.data = []

        for fp  in datafp:
            self.data.append(self.load_data(logger, fp))

        self.length = sum([t[0].shape[0] for t in self.data])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        quo, rem = divmod(idx, self.n_files)
        return self.data[rem][0][quo], self.data[rem][1][quo], \
            self.data[rem][2][quo], self.data[rem][3][quo], \
            self.data[rem][4][quo]

    def load_data(self, logger, fp):
        with open(fp+'_dijkstra.pkl', 'rb') as f:
            dijkstra_steps_numpy = pkl.load(f)

        dijkstra_steps = [(torch.tensor(t0), torch.tensor(t1), torch.tensor(t2)) for t0, t1, t2 in dijkstra_steps_numpy]
        adj, weights,_ = dijkstra_steps[-1]
        dijkstra_state, pred, dijkstra_term = zip(*(dijkstra_steps[:-1]))
        dijkstra_state_tensor = torch.stack(dijkstra_state, dim=-1)
        pred_tensor = torch.stack(pred, dim=-1)
        term_tensor = torch.stack(dijkstra_term, dim=-1)

        return adj, weights, dijkstra_state_tensor, pred_tensor, term_tensor


def collate_dijkstra(batch):
    adj = torch.stack([item[0] for item in batch], dim=0)
    weights = torch.stack([item[1] for item in batch], dim=0)
    dijkstra_state = torch.stack([item[2] for item in batch], dim=0)
    pred = torch.stack([item[3] for item in batch], dim=0)
    term = torch.stack([item[4] for item in batch], dim=0)
    return adj, weights, dijkstra_state, pred, term

class DFSSteps(Dataset):
    """
    Important!! All graphs must be of the size in a given dataset!
    """

    def __init__(self, logger, datafp, name='Train'):
        """
        logger: for printing out information about the dataset
        datafp: list of one or more strings representing filepaths to data to be loaded
        """
        self.log = logger
        self.name = name
        self.n_files = len(datafp)
        if self.n_files < 1:
            logger.error("No filepaths have been passed to the ShortestPathSteps dataset constructor!")

        self.data = []

        for fp  in datafp:
            self.data.append(self.load_data(logger, fp))

        self.length = sum([t[0].shape[0] for t in self.data])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        quo, rem = divmod(idx, self.n_files)
        return self.data[rem][0][quo], \
            self.data[rem][1][quo], self.data[rem][2][quo]

    def load_data(self, logger, fp):
        with open(fp+'_dfs.pkl', 'rb') as f:
            dfs_steps_numpy = pkl.load(f)

        dfs_steps = [(torch.tensor(t0), torch.tensor(t1)) for t0, t1 in dfs_steps_numpy]
        adj, _ = dfs_steps[-1]
        dfs_state, dfs_term = zip(*(dfs_steps[:-1]))
        dfs_state_tensor = torch.stack(dfs_state, dim=-1)
        term_tensor = torch.stack(dfs_term, dim=-1)

        return adj, dfs_state_tensor, term_tensor


def collate_dfs(batch):
    adj = torch.stack([item[0] for item in batch], dim=0)
    weights = None
    dfs_state = torch.stack([item[1] for item in batch], dim=0)
    term = torch.stack([item[2] for item in batch], dim=0)
    return adj, weights, dfs_state, term

class TopoSortSteps(Dataset):
    """
    Data produced by Prim's algorithm, miminum spanning tree task task on
    various graphs including each step of the algorithm as well as solving the
    shortest path problem simultaeneously.

    Important!! All graphs must be of the size in a given dataset!
    """

    def __init__(self, logger, datafp, name='Train'):
        """
        logger: for printing out information about the dataset
        datafp: list of one or more strings representing filepaths to data to be loaded
        """
        self.log = logger
        self.name = name
        self.n_files = len(datafp)
        if self.n_files < 1:
            logger.error("No filepaths have been passed to the ShortestPathSteps dataset constructor!")

        self.data = []

        for fp  in datafp:
            self.data.append(self.load_data(logger, fp))

        self.length = sum([t[0].shape[0] for t in self.data])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        quo, rem = divmod(idx, self.n_files)
        return self.data[rem][0][quo], self.data[rem][1][quo], \
            self.data[rem][2][quo], self.data[rem][3][quo]

    def load_data(self, logger, fp):
        with open(fp+'_toposort.pkl', 'rb') as f:
            steps_numpy = pkl.load(f)

        toposort_steps = [(torch.tensor(t0), torch.tensor(t1), torch.tensor(t2)) for t0, t1, t2 in steps_numpy]
        adj, _ ,_ = toposort_steps[-1]
        toposort_state, pred, toposort_term = zip(*(toposort_steps[:-1]))
        toposort_state_tensor = torch.stack(toposort_state, dim=-1)
        pred_tensor = torch.stack(pred, dim=-1)
        term_tensor = torch.stack(toposort_term, dim=-1)

        return adj, toposort_state_tensor, pred_tensor, term_tensor


def collate_toposort(batch):
    adj = torch.stack([item[0] for item in batch], dim=0)
    toposort_state = torch.stack([item[1] for item in batch], dim=0)
    pred = torch.stack([item[2] for item in batch], dim=0)
    term = torch.stack([item[3] for item in batch], dim=0)
    return adj, None, toposort_state, pred, term

class DFS2Steps(Dataset):
    """
    Data produced by Prim's algorithm, miminum spanning tree task task on
    various graphs including each step of the algorithm as well as solving the
    shortest path problem simultaeneously.

    Important!! All graphs must be of the size in a given dataset!
    """

    def __init__(self, logger, datafp, name='Train'):
        """
        logger: for printing out information about the dataset
        datafp: list of one or more strings representing filepaths to data to be loaded
        """
        self.log = logger
        self.name = name
        self.n_files = len(datafp)
        if self.n_files < 1:
            logger.error("No filepaths have been passed to the ShortestPathSteps dataset constructor!")

        self.data = []

        for fp  in datafp:
            self.data.append(self.load_data(logger, fp))

        self.length = sum([t[0].shape[0] for t in self.data])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        quo, rem = divmod(idx, self.n_files)
        return self.data[rem][0][quo], self.data[rem][1][quo], \
            self.data[rem][2][quo], self.data[rem][3][quo]

    def load_data(self, logger, fp):
        with open(fp+'_dfs2.pkl', 'rb') as f:
            steps_numpy = pkl.load(f)

        steps = [(torch.tensor(t0), torch.tensor(t1), torch.tensor(t2)) for t0, t1, t2 in steps_numpy]
        adj, _ ,_ = steps[-1]
        state, pred, term = zip(*(steps[:-1]))
        state_tensor = torch.stack(state, dim=-1)
        pred_tensor = torch.stack(pred, dim=-1)
        term_tensor = torch.stack(term, dim=-1)

        return adj, state_tensor, pred_tensor, term_tensor


def collate_dfs2(batch):
    adj = torch.stack([item[0] for item in batch], dim=0)
    state = torch.stack([item[1] for item in batch], dim=0)
    pred = torch.stack([item[2] for item in batch], dim=0)
    term = torch.stack([item[3] for item in batch], dim=0)
    return adj, None, state, pred, term

class PrimsDijkstraSteps(Dataset):
    """
    Important!! All graphs must be of the size in a given dataset!
    """

    def __init__(self, logger, datafp, name='Train'):
        """
        logger: for printing out information about the dataset
        datafp: list of one or more strings representing filepaths to data to be loaded
        """
        self.log = logger
        self.name = name
        self.n_files = len(datafp)
        if self.n_files < 1:
            logger.error("No filepaths have been passed to the ShortestPathSteps dataset constructor!")

        self.data = []

        for fp  in datafp:
            self.data.append(self.load_data(logger, fp))

        self.length = sum([t[0].shape[0] for t in self.data])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        quo, rem = divmod(idx, self.n_files)
        return self.data[rem][0][quo], self.data[rem][1][quo], \
            self.data[rem][2][quo], self.data[rem][3][quo], \
            self.data[rem][4][quo]

    def load_data(self, logger, fp):
        with open(fp+'_dijkstra.pkl', 'rb') as f:
            dijkstra_steps_numpy = pkl.load(f)

        dijkstra_steps = [(torch.tensor(t0), torch.tensor(t1), torch.tensor(t2)) for t0, t1, t2 in dijkstra_steps_numpy]
        adj, weights,_ = dijkstra_steps[-1]
        dijkstra_state, pred, dijkstra_term = zip(*(dijkstra_steps[:-1]))
        dijkstra_state_tensor = torch.stack(dijkstra_state, dim=-1)
        dijkstra_pred_tensor = torch.stack(pred, dim=-1)
        term_tensor = torch.stack(dijkstra_term, dim=-1)

        with open(fp+'_prims.pkl', 'rb') as f:
            prims_steps_numpy = pkl.load(f)

        prims_steps = [(torch.tensor(t0), torch.tensor(t1), torch.tensor(t2)) for t0, t1, t2 in prims_steps_numpy]
        prims_adj, prims_weights,_ = prims_steps[-1]
        if not torch.equal(prims_adj, adj):
            self.log.exception("The graphs for prims and dijkstra are different. Erroneous file: " + fp)
        if not torch.equal(prims_weights, weights):
            self.log.exception("The weights for prims and dijkstra are different. Erroneous file: " + fp)
        prims_state, pred, prims_term = zip(*(prims_steps[:-1]))
        prims_state_tensor = torch.stack(prims_state, dim=-1)
        prims_pred_tensor = torch.stack(pred, dim=-1)
        prims_term_tensor = torch.stack(prims_term, dim=-1)
        # we are done when the slower algorithm is done
        term_tensor = torch.min(term_tensor, prims_term_tensor)

        #stack the pred tensors
        pred_tensor = torch.stack([prims_pred_tensor, dijkstra_pred_tensor], dim=-1)

        #stack the state tensors
        state_tensor = torch.cat([prims_state_tensor.unsqueeze(2), dijkstra_state_tensor], dim=2)

        return adj, weights, state_tensor, pred_tensor, term_tensor


def collate_primsdijkstra(batch):
    adj = torch.stack([item[0] for item in batch], dim=0)
    weights = torch.stack([item[1] for item in batch], dim=0)
    state = torch.stack([item[2] for item in batch], dim=0)
    pred = torch.stack([item[3] for item in batch], dim=0)
    term = torch.stack([item[4] for item in batch], dim=0)
    return adj, weights, state, pred, term

class Distilled(Dataset):
    """
    Important!! All graphs must be of the size in a given dataset!
    """

    def __init__(self, logger, batched_output, name='Distilled'):
        """
        logger: for printing out information about the dataset
        datafp: list of one or more strings representing filepaths to data to be loaded
        """
        self.log = logger
        self.name = name

        self.data = batched_output
        self.bsize = self.data[0][0].shape[0]
        self.elements = len(self.data[0])

        self.length = sum([t[0].shape[0] for t in self.data])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        quo, rem = divmod(idx, self.bsize)
        item = [self.data[quo][i][rem] for i in range(self.elements)]
        return tuple(item)

def collate_distilled(batch):
    stacked_batch = []
    for i in range(len(batch[0])):
        stacked_batch.append(torch.stack([item[i] for item in batch], dim=0))
    return tuple(stacked_batch)
