import dataloaders as dl
from torch.utils.data import DataLoader

def multi_stream(ngraph_train, ngraph_val, nnode, logger, algo_names,
                 ngraph_test:list, nnode_test:list, graph='erdosrenyi'):
    train_datafp = 'Data/train_%s%s_%s' % (graph, ngraph_train, nnode)
    val_datafp = 'Data/val_%s%s_%s' % (graph, ngraph_val, nnode)
    test_datafp = ['Data/test_%s%s_%s' % (graph, ngraph_test[i], nnode_test[i]) for i in range(len(nnode_test))]

    batchsize = 10

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