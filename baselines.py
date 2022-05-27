from loss import LossAssembler, create_loss_class
from train import train
import evaluate as ev

def train_multi(model, logger, task_list, train_stream, val_stream, train_params, test_stream, device='cpu'):
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

def train_multi(model, logger, task_list, train_stream, val_stream, train_params, test_stream, device='cpu'):
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