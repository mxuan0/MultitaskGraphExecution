import argparse
import configparser
import logging
import sys
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config as cfg
from datagen.algorithms import info, gen_multi_algo_data
from datagen.graphgen import gen_erdos_renyi, gen_barabasi_albert, gen_twod_grid
import dataloaders as dl
from loss import LossAssembler, create_loss_class
import model as arch
import initialisation as init
from train import train, train_metadata
import evaluate as ev
import pdb
def main():
    ## read config and cmd line arguments & set up logging
    args, logger = pre_main()
    args = vars(args)

    # determine cuda device
    device = torch.device('cuda' if args['cuda'] and torch.cuda.is_available() else 'cpu')

    # for reproducibility set the seed
    torch.manual_seed(args['seed'])

    '''if args['runtests']:
        test.run(logger)
        logger.info("Done with tests")
        sys.exit(0)'''

    transfer = False
    if (args['transfertask'] != '') and (args['transfertask'] != 'None'):
        transfer = True
        transfertasks = args['transfertask'].split(' ')
        transfer_algo_names = list(map(lambda x: x.split('_')[-1], transfertasks))
        transfer_metadata = info(logger, transfer_algo_names)

    # getting metadata about the task
    task_list = args['task'].split(' ')
    algo_names = list(map(lambda x: x.split('_')[-1], task_list))
    metadata = info(logger, algo_names)
    logger.info("Task metadata: " + str(metadata))
    pdb.set_trace()
    if args['datagen']:
        rand_gen = torch.distributions.Uniform(0.0, 1.0)
        ngraphs = args['ngraphs'].split(',')
        nnodes = args['nnodes'].split(',')
        #filename = args['datagraphsetname'].split(',')
        for graphtype, params in zip(args['graphtypes'].split(','), args['graphparams'].split(',')):
            for i in range(len(ngraphs)):
                logger.info("Generating {} graphs".format(graphtype))
                if graphtype == 'ErdosRenyi':
                    directed = True if params == "True" else False
                    gen_erdos_renyi(rand_gen, int(ngraphs[i]), int(nnodes[i]), args['datasavefp']+filename[i]+"_", directed)
                    src_nodes = torch.argmax(rand_gen.sample((int(ngraphs[i]), int(nnodes[i]))), dim=1)
                    gen_multi_algo_data(
                        args['datasavefp']+filename[i]+"_"+"erdosrenyi" + ngraphs[i] + "_"+ nnodes[i] + ".pt",
                        src_nodes,
                        task_list,
                        args['hidekeys']
                    )
                elif graphtype == "BarabasiAlbert":
                    m = int(params)
                    gen_barabasi_albert(rand_gen, int(ngraphs[i]), int(nnodes[i]), m, args['datasavefp']+filename[i]+"_")
                    src_nodes = torch.argmax(rand_gen.sample((int(ngraphs[i]), int(nnodes[i]))), dim=1)
                    gen_multi_algo_data(
                        args['datasavefp']+filename[i]+"_"+"barabasialbert" + ngraphs[i] + "_"+ nnodes[i] + ".pt",
                        src_nodes,
                        task_list,
                        args['hidekeys']
                    )
                elif graphtype == "TwoDGrid":
                    gen_twod_grid(rand_gen, int(ngraphs[i]), int(nnodes[i]), args['datasavefp']+filename[i]+"_")
                    src_nodes = torch.argmax(rand_gen.sample((int(ngraphs[i]), int(nnodes[i]))), dim=1)
                    gen_multi_algo_data(
                        args['datasavefp']+filename[i]+"_"+"twodgrid" + ngraphs[i] + "_"+ nnodes[i] + ".pt",
                        src_nodes,
                        task_list,
                        args['hidekeys']
                    )
                else:
                    logger.exception("Unknown graphtype {}")
        logger.info("Data generation done")

    # collect relevant stuff for training
    dset = dl.MultiAlgo
    collate_batch = dl.collate_multi_algo
    metrics = [m for t in task_list for m in ev.get_metrics(t)]
    # loss_train, loss_val, loss_test, metrics, dset, collate_batch = loss_and_data_fns(logger, task_list)

    algos = []
    for task in task_list:
        algos.append(
            create_loss_class(task, args)
        )
    loss_module = LossAssembler(device, logger, algos, args)

    # normalise the training loss or not
    cfg.normaliseloss = args['normaliseloss']

    # loading train data
    data_stream = DataLoader(dset(logger,args['datafp'].split(' '),algo_names,"Train"),
                             shuffle = True,
                             batch_size = args['batchsize'],
                             num_workers = args['nworkers'],
                             collate_fn = collate_batch,
                             drop_last = False
                             )
    # loading val data
    val_stream = DataLoader(dset(logger,args['valdatafp'].split(' '),algo_names,"Validation"),
                            shuffle = False,
                            batch_size = args['batchsize'],
                            num_workers = args['nworkers'],
                            collate_fn = collate_batch,
                            drop_last = False
                            )
    # loading test data
    test_stream = []
    for task, fp in zip(args['evaltask'].split(','),args['evaldatafp'].split(' ')):
        test_stream.append(DataLoader(dset(logger,[fp],algo_names,task),
                                      shuffle = False,
                                      batch_size = args['evalbatchsize'],
                                      num_workers = args['nworkers'],
                                      collate_fn = collate_batch,
                                      drop_last = False
                                      )
                           )
    logger.info("Dataloaders initialised")

    # potentially hoist model creation code to function when adding more models
    model = arch.create_model(args['arch'], device, args, metadata)
    train_params = {k:args[k] for k in train_metadata()}

    if args['optimal']:
        logger.info("Using optimal model")
        model = arch.optimal_model(device, logger, metadata, task_list)

    logger.info("Model initialised")
    if args['loadmodel'] != '':
        logger.info("Loading model from " + args['loadmodel'])
        checkpoint = torch.load(args['loadmodel'], map_location=torch.device('cuda') if args['cuda'] else torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state'])
    else:
        recorder = None
        # creating the training parameter dict
        model_state, val_loss = train(logger,
                                      device,
                                      data_stream,
                                      val_stream,
                                      model,
                                      train_params,
                                      loss_module,
                                      recorder
                                      )
        if args['savemodel'] != '':
            torch.save({'model_state': model_state}, args['savemodel'])

    # evaluation, i.e. test time loss + fancy stuff
    ev.evaluate(logger,
                device,
                test_stream,
                model,
                loss_module,
                metrics
                )

    if transfer:
        ## it is currently assumed that at transfer time we always use the no_algo loss
        ## to change that in the future register them as separate tasks in the future
        import no_algo_loss as loss
        # collect relevant stuff for training
        dset = dl.MultiAlgo
        collate_batch = dl.collate_multi_algo
        metrics = [m for t in transfertasks for m in ev.get_metrics(t)]
        # loss_train, loss_val, loss_test, metrics, dset, collate_batch = loss_and_data_fns(logger, task_list)

        algos = []
        for task in transfertasks:
            algos.append(
                create_loss_class(task, args)
            )
        loss_module = LossAssembler(device, logger, algos, args)

        # loading train data
        data_stream = DataLoader(dset(logger,args['datafp'].split(' '),transfer_algo_names,"Train"),
                                 shuffle = True,
                                 batch_size = args['batchsize'],
                                 num_workers = args['nworkers'],
                                 collate_fn = collate_batch,
                                 drop_last = False
                                 )
        # loading val data
        val_stream = DataLoader(dset(logger,args['valdatafp'].split(' '),transfer_algo_names,"Validation"),
                                shuffle = False,
                                batch_size = args['batchsize'],
                                num_workers = args['nworkers'],
                                collate_fn = collate_batch,
                                drop_last = False
                                )
        # loading test data
        test_stream = []
        for task, fp in zip(args['evaltask'].split(','),args['evaldatafp'].split(' ')):
            test_stream.append(DataLoader(dset(logger,[fp],transfer_algo_names,task),
                                          shuffle = False,
                                          batch_size = args['evalbatchsize'],
                                          num_workers = args['nworkers'],
                                          collate_fn = collate_batch,
                                          drop_last = False
                                          )
                               )
        arch_type = args['transferarch'] if args['transferarch'] != '' else args['arch']
        transfer_model = arch.create_model(arch_type, device, args, transfer_metadata)
        model_state = model.state_dict()
        init.processor_init(transfer_model, model_state)
        train_params['lr'] = args['tunelr']
        _ , val_loss = train(logger,
                            device,
                            data_stream,
                            val_stream,
                            transfer_model,
                            train_params,
                            loss_module
                            )
        target_model = transfer_model
        ev.evaluate(logger,
                    device,
                    test_stream,
                    target_model,
                    loss_module,
                    metrics
                    )

    return None


### boilerplate code for logging and commandline arguments
def pre_main():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    ## parse the config file
    conf_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )
    conf_parser.add_argument('-c', '--config',
                             default='./options.conf',
                             help='Path to config file',
                             metavar="FILE"
    )
    args, remaining_argv = conf_parser.parse_known_args()

    # default values as specified by the config file
    defaults = {}

    '''if args.config:
        config = configparser.ConfigParser()
        config.read([args.config])
        for key, val in config.items('Defaults'):
            if val == 'True' or val == 'False':
                defaults[key] = config.getboolean('Defaults', key)
            else:
                defaults[key] = val'''

    # parse the remaining arguments
    # this Parser also generates the help message
    parser = argparse.ArgumentParser(
        # inherit options from config_parser
        parents=[conf_parser]
    )
    # add all the command line arguments here
    parser.add_argument('-d', '--dataset',
                        type=str,
                        default='',
                        help='Dataset to use'
    )
    parser.add_argument('-f', '--datafp',
                        type=str,
                        default='./data',
                        help='Path to dataset'
    )
    parser.add_argument('--valdatafp',
                        type=str,
                        default='./data/val',
                        help='Path to validation dataset'
    )
    parser.add_argument('--evaldatafp',
                        type=str,
                        default='./data/eval',
                        help='Path to test dataset'
    )
    parser.add_argument('--evaltask',
                        type=str,
                        default='Test',
                        help='Name of test data'
    )
    parser.add_argument('--transfertask',
                        type=str,
                        default='',
                        help='List of tasks to test transfer of algorithmic alignment on. Empy List means no transfer testing.'
    )
    parser.add_argument('--transferarch',
                        type=str,
                        default='',
                        help='Transfer architecture to use'
    )
    parser.add_argument('--datasavefp',
                        type=str,
                        default='./',
                        help='Path to save dataset'
    )
    parser.add_argument('--savemodel',
                        type=str,
                        default='',
                        help='Path to save model checkpoint, ignored if empty string.'
    )
    parser.add_argument('--loadmodel',
                        type=str,
                        default='',
                        help='Path to load the model checkpoint from, ignored if empty string'
    )
    parser.add_argument('--graphtypes',
                        type=str,
                        default='ErdosRenyi',
                        help='The graphtype to generate if datagen is set to True. Options are: ErdosRenyi, BarabasiAlbert'
    )
    parser.add_argument('--graphparams',
                        type=str,
                        default='False',
                        help='The graphparams for the graphtypes once per graphsetname. ' + \
                             'For ErdosRenyi: directed (bool), BarabasiAlbert: m (int)'
    )
    parser.add_argument('--datagen',
                        type=str2bool,
                        default=False,
                        help='Generate data'
    )
    parser.add_argument('--hidekeys',
                        type=str2bool,
                        default=True,
                        help='Generate data, while hiding the keys'
    )
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='If verbose flag is set then all the logging output goes to std_out as well'
    )
    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        help='If debug flag is set then debug messages be printed'
    )
    parser.add_argument('-l', '--logfile',
                        type=str,
                        default='./last_run.log',
                        help='File path to logfile'
    )
    parser.add_argument('--cuda',
                        type=str2bool,
                        default=torch.cuda.is_available(),
                        help='Decides whether to use GPU or CPU'
    )
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Torch seed'
    )
    parser.add_argument('--optimal',
                        type=str2bool,
                        default=False,
                        help='Use optimal parameters for pre-determined encoding dim, not all tasks have them'
    )
    parser.add_argument('-t', '--task',
                        type=str,
                        default='',
                        help='The task to run on.'
    )
    parser.add_argument('-a', '--arch',
                        type=str,
                        default='NeuralExec',
                        help='The architecture to use for the model. Current options are: NeuralExec, NCNeuralExec.'
    )
    parser.add_argument('--layers',
                        type=int,
                        default=1,
                        help='Number of layers for the GNN arch'
    )
    # train_params
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        help='Optimizer choice'
    )
    parser.add_argument('--epochs',
                        type=int,
                        default=0,
                        help='Number of max epochs for training'
    )
    parser.add_argument('--ksamples',
                        type=int,
                        default=1,
                        help='Number of trajectories to sample'
    )
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='Learning rate'
    )
    parser.add_argument('--weightdecay',
                        type=float,
                        default=0.00001,
                        help='L2 regularisation also called weightdecay'
    )
    parser.add_argument('--warmup',
                        type=int,
                        default=0,
                        help='How many warm-up epochs'
    )
    parser.add_argument('--earlystop',
                        type=str2bool,
                        default=True,
                        help='Whether or not to do Early Stopping'
    )
    parser.add_argument('--earlytol',
                        type=float,
                        default=5e-6,
                        help='The tolerance for the early stopping criterion'
    )
    parser.add_argument('--patience',
                        type=int,
                        default=1,
                        help='Patience during early stopping'
    )
    parser.add_argument('--schedpatience',
                        type=int,
                        default=0,
                        help='Scheduler patience before reducing lr'
    )
    parser.add_argument('--tempinit',
                        type=float,
                        default=1.0,
                        help='Initial temperature for softmax'
    )
    parser.add_argument('--temprate',
                        type=float,
                        default=1.0,
                        help='Temperature rate (multiplier)'
    )
    parser.add_argument('--tempmin',
                        type=float,
                        default=1.0,
                        help='Temperature minium value'
    )
    parser.add_argument('--normaliseloss',
                        type=str2bool,
                        default=False,
                        help='Whether or not to normalise the loss to log(#nodes)'
    )
    # transfer params
    parser.add_argument('--iterations',
                        type=int,
                        default=1,
                        help='Number of transfer exploit and explore iterations'
    )
    parser.add_argument('--transferepochs',
                        type=int,
                        default=1,
                        help='Number of transfer exploit and explore iterations'
    )
    parser.add_argument('--tunelr',
                        type=float,
                        default=0.0001,
                        help='Learning rate'
    )
    # model parameters
    parser.add_argument('--latentdim',
                        type=int,
                        default=32,
                        help='Size of the latent dimension in the processor network'
    )
    parser.add_argument('--encdim',
                        type=int,
                        default=32,
                        help='Size of the encoding dimension in the encoder network'
    )
    parser.add_argument('--noisedim',
                        type=int,
                        default=0,
                        help='Size of to inject for each node, this serves to help the GNN identify nodes uniquely'
    )
    # dataloader a params
    parser.add_argument('--batchsize',
                        type=int,
                        default=32,
                        help='Size of batches'
    )
    parser.add_argument('--evalbatchsize',
                        type=int,
                        default=32,
                        help='Size of eval batches'
                        )
    parser.add_argument('--nworkers',
                        type=int,
                        default=0,
                        help='Number of workers in the dataloader'
    )
    # datagen params
    parser.add_argument('--ngraphs',
                        type=str,
                        default='',
                        help='List of number of graphs'
    )
    parser.add_argument('--nnodes',
                        type=str,
                        default='',
                        help='List of number of nodes per graph'
    )
    parser.add_argument('--datagraphsetnam',
                        type=str,
                        default='',
                        help='List of number of nodes per graph'
    )

    # note: set_default won't work with any variables with dashes/underscores in the name
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)

    ## set up logging
    logFormatter = logging.Formatter(
        '%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s'
    )
    rootLogger = logging.getLogger()
    # set debugging level
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    rootLogger.setLevel(level)

    # file logging to logfile
    fileHandler = logging.FileHandler(args.logfile)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # logging to std out
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    # if verbose is set we print level else we print warnings only
    if args.verbose:
        consoleHandler.setLevel(level)
    else:
        consoleHandler.setLevel(logging.WARNING)
    rootLogger.addHandler(consoleHandler)

    # print options to log
    rootLogger.info(args)

    return args, rootLogger

if __name__ == '__main__':
    main()
