from loss import LossAssembler, create_loss_class
from train import train, train_seq_reptile
import evaluate as ev

def run_multi(model, logger, task_list, train_stream, val_stream, train_params, test_stream, device='cpu'):
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

def run_seq_reptile(model, logger, task_list, train_stream, val_stream, train_params, test_stream, device='cpu'):
    
    loss_module_dict = {}
    for task in task_list:
        algo = [create_loss_class(task, {'hidekeys':True})]
        loss_module_dict[task] = LossAssembler(device, logger, algo, {'ksamples':1})

    recorder = None
    # creating the training parameter dict
    model_state, val_loss = train_seq_reptile(logger,
                                    device,
                                    train_stream,
                                    val_stream,
                                    model,
                                    train_params,
                                    loss_module_dict,
                                    )

    # metrics = [m for t in task_list for m in ev.get_metrics(t)]
    # ev.evaluate(logger,
    #             device,
    #             test_stream,
    #             model,
    #             loss_module,
    #             metrics
    #             )