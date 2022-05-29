import dataloaders as dl
from torch.utils.data import DataLoader, RandomSampler
import collections, pdb

def multi_stream(ngraph_train, ngraph_val, nnode, logger, algo_names,
                 ngraph_test:list, nnode_test:list, graph='erdosrenyi', batchsize=10):
    train_datafp = 'Data/train_%s%s_%s' % (graph, ngraph_train, nnode)
    val_datafp = 'Data/val_%s%s_%s' % (graph, ngraph_val, nnode)
    test_datafp = ['Data/test_%s%s_%s' % (graph, ngraph_test[i], nnode_test[i]) for i in range(len(nnode_test))]

    dset = dl.MultiAlgo

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
    return train_stream, val_stream, test_stream

algo_to_dataset = {
    'bfs' : dl.ReachabilitySteps,
    'bf' : dl.BFSteps
}
algo_to_collate = {
    'bfs' : dl.collate_reach,
    'bf' : dl.collate_bf
}

def seq_reptile_stream(ngraph_train:list, ngraph_val, nnode, logger, algo_names,
                 ngraph_test:list, nnode_test:list, graph='erdosrenyi', batchsize=10):
    train_datafp = ['Data/train_%s%s_%s' % (graph, ngraph, nnode) for ngraph in ngraph_train]
    val_datafp = 'Data/val_%s%s_%s' % (graph, ngraph_val, nnode)
    test_datafp = ['Data/test_%s%s_%s' % (graph, ngraph_test[i], nnode_test[i]) for i in range(len(nnode_test))]

    train_stream = {}
    for i in range(len(algo_names)):
        algo = algo_names[i]
        ds = algo_to_dataset[algo](logger,train_datafp[i].split(' '),"Train")
        sampler = RandomSampler(ds, replacement=True)
        train_stream[algo] = DataLoader(ds,
                                    #shuffle = True,
                                    batch_size = batchsize,
                                    sampler=sampler,
                                    collate_fn = algo_to_collate[algo],
                                    drop_last = False
                                    )
        #pdb.set_trace()
    val_stream = {}
    for algo in algo_names:
        ds = algo_to_dataset[algo](logger,val_datafp.split(' '),"Validation")
        val_stream[algo] = DataLoader(ds,
                                    shuffle = False,
                                    batch_size = batchsize,
                                    collate_fn = algo_to_collate[algo],
                                    drop_last = False
                                    )
                                
    test_stream = collections.defaultdict(list)
    for fp in test_datafp:
        for algo in algo_names:
            ds = algo_to_dataset[algo](logger,[fp],'Test')
            test_stream[algo].append(DataLoader(ds,
                                            shuffle = False,
                                            batch_size = batchsize,
                                            collate_fn = algo_to_collate[algo],
                                            drop_last = False
                                            )
                            )
    return train_stream, val_stream, test_stream