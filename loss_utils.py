import config as cfg

import torch
import torch.nn as nn
import torch.nn.functional as fn

### Multiple trajectories helpers
def max_of_k_samples(losses, batch_size, k):
    """
    if k is positive we take the min, if k is negative we take the mean
    """
    max_losses = []
    for loss in losses:
        if loss.ndim == 0:
            max_losses.append(loss)
        else:
            if k > 0:
                max_losses.append(
                    torch.min(loss.unsqueeze(-1).view(batch_size, k, -1), dim=1)[0]
                )
            else:
                max_losses.append(
                    torch.mean(loss.unsqueeze(-1).view(batch_size, abs(k), -1), dim=1)
                )
    return list(max_losses)

### Loss utilities
def state_loss(y, target, term_mask):
    loss_state_steps = fn.binary_cross_entropy_with_logits(y,
                                                           target,
                                                           reduction='none'
                                                           )
    loss_state_steps = term_mask.unsqueeze(1) * loss_state_steps
    return  torch.mean(loss_state_steps, dim=1)

# Normalise want a random guess to be roughly unit magnitude loss
def pred_loss(device, p, target, pred_mask, term_mask, node_mask):
    """
    this function always assumes that if mask exists that the node itself is a valid predecessor
    """
    bsize = p.shape[0]
    nnodes = p.shape[1]

    if pred_mask is not None:
        mask = pred_mask.unsqueeze(1).expand_as(p) \
            + torch.eye(nnodes,
                        device=device
                        ).unsqueeze(0).expand(bsize, -1, -1).bool()
        p = torch.where(
            mask,
            p.double(),
            float('-inf')
        ).float()
    loss_pred_steps = fn.cross_entropy(p.reshape(-1,nnodes),
                                    target.reshape(-1).long(),
                                    reduction='none'
                                    ).view(bsize,nnodes)
    # mask both unreachable nodes + mask examples that have terminated
    loss_pred_steps = term_mask.unsqueeze(1) * (node_mask) * loss_pred_steps
    div = torch.maximum((node_mask).float().sum(dim=1), torch.ones((bsize,), device=device))
    return torch.sum(loss_pred_steps, dim=1)/div

# Normalise want a random guess to be roughly unit magnitude loss
def soft_pred_loss(device, p, target, term_mask, node_mask):
    """
    this function always assumes that if mask exists that the node itself is a valid predecessor
    """
    bsize = p.shape[0]
    nnodes = p.shape[1]

    log_prob = fn.log_softmax(p.reshape(-1, nnodes), dim=-1)
    log_prob = torch.where(~torch.isinf(log_prob),
                           log_prob,
                           torch.tensor([0.], device=device)
                           )
    loss_pred_steps = (-(target.reshape(-1, nnodes)*log_prob).sum(dim=-1)).view(bsize,nnodes)
    # mask both unreachable nodes + mask examples that have terminated
    loss_pred_steps = term_mask.unsqueeze(1) * (node_mask) * loss_pred_steps
    div = torch.maximum((node_mask).float().sum(dim=1), torch.ones((bsize,), device=device))
    return torch.sum(loss_pred_steps, dim=1)/div

def term_loss(tau, term, pos_w, term_mask):
    return term_mask * fn.binary_cross_entropy_with_logits(tau,
                                               term.float().unsqueeze(1),
                                               reduction='none',
                                               # pos_weight=pos_w
                                               ).squeeze()

def next_node_loss(y, target, term_mask):
    y = torch.where(term_mask.unsqueeze(1).expand(-1, y.shape[1]).bool(),
                           y,
                           torch.tensor([0.], device=y.device)
                           ).float()
    # y_target = torch.gather(y, 1, target)
    # y_target = torch.where(torch.isinf(y_target),
    #                        torch.tensor([0.], device=y.device),
    #                        y_target
    #                        )
    # y = torch.scatter(y,
    #                   1,
    #                   target.unsqueeze(1),
    #                   0
    #                   ).float()
    loss = fn.cross_entropy(y,
                            target,
                            reduction='none'
                            )
    loss = term_mask * loss
    return loss

def soft_next_node_loss(y, target, term_mask):
    y = torch.where(term_mask.unsqueeze(1).expand(-1, y.shape[1]).bool(),
                           y,
                           torch.tensor([0.], device=y.device)
                           ).float()
    log_prob = fn.log_softmax(y, dim=1)
    loss = -(target*log_prob).sum(dim=-1)
    loss = term_mask * loss
    return loss

def dist_loss(y, target, mask, term_mask):
    loss_dist_steps = fn.mse_loss(y,
                                target,
                                reduction='none'
                                )
    if cfg.normaliseloss:
        var_target = torch.var(target, dim=1, unbiased=False, keepdim=True)
        loss_dist_steps = term_mask * (mask) * loss_dist_steps * (1/var_target) \
            * torch.log(torch.tensor([float(y.shape[1])], device=y.device))
    else:
        loss_dist_steps = term_mask * (mask) * loss_dist_steps
    return torch.sum(loss_dist_steps, dim=1)/((mask).float().sum(dim=1))

def l1_dist_loss(y, target, mask, term_mask):
    loss_dist_steps = fn.smooth_l1_loss(y,
                                        target,
                                        beta=0.001,
                                        reduction='none'
                                        )
    if cfg.normaliseloss:
        var_target = torch.var(target, dim=1, unbiased=False, keepdim=True)
        loss_dist_steps = term_mask * (mask) * loss_dist_steps * (1/var_target) \
            * torch.log(torch.tensor([float(y.shape[1])], device=y.device))
    else:
        loss_dist_steps = term_mask * (mask) * loss_dist_steps
    div = (torch.max((mask).float().sum(dim=1),torch.tensor([1.],device=y.device)))
    return torch.sum(loss_dist_steps, dim=1)/div
### Test utilities

def get_laststep(tau):
    sig_tau = torch.sigmoid(tau)>0.5
    sig_tau[:,:,-1] = 1
    # the following slightly mad line of code, simply tries to determine the index of the first time tau predicts to terminate
    last_step = (torch.cumsum(torch.cumsum(sig_tau,dim=-1),dim=-1)==1).nonzero(as_tuple=False).squeeze(dim=-1)[:,-1]
    return last_step

def next_node_test(y, target, term_mask):
    loss_prim_steps = (y == target).long()
    loss_prim_steps = (term_mask) * loss_prim_steps
    loss_prim_steps = loss_prim_steps.sum(dim=-1)/(term_mask).sum(dim=-1).squeeze()
    return loss_prim_steps.sum()

def pred_test_mean(device, p, target, pred_mask, term_mask, node_mask):
    """
    this function always assumes that if mask exists that the node itself is a valid predecessor
    """
    bsize = p.shape[0]
    nnodes = p.shape[1]
    n_steps = term_mask.shape[-1]

    if pred_mask is not None:
        mask = pred_mask[:,:,:n_steps].unsqueeze(1).transpose(-1,-2).expand_as(p[:,:,:n_steps,:]) \
            + torch.eye(nnodes,
                        device=device
                        ).unsqueeze(0).unsqueeze(2).expand_as(p[:,:,:n_steps, :]).bool()
        p[:,:,:n_steps,:] = torch.where(
            mask,
            p[:,:,:n_steps,:].double(),
            float('-inf')
        ).float()

    loss_pred_steps = (torch.argmax(p[:,:,:n_steps,:], dim=-1) == target).long()
    loss_pred_steps =  (term_mask).unsqueeze(1) * node_mask * loss_pred_steps
    div = torch.maximum((node_mask).float().sum(dim=1), torch.ones((bsize,n_steps), device=device))
    loss_pred_steps = torch.sum(loss_pred_steps, dim=1)/div
    loss_pred_steps = loss_pred_steps.sum(dim=-1)/(term_mask).sum(dim=-1).squeeze()
    return loss_pred_steps.sum()

def term_test(tau, term, term_mask):
    loss_tau_steps = ((torch.sigmoid(tau)>0.5) == term).long()
    loss_tau_steps = (term_mask) * loss_tau_steps
    loss_tau_steps = loss_tau_steps.sum(dim=-1).float()/(term_mask).sum(dim=-1).squeeze()
    return loss_tau_steps.sum()

def pred_test_last(device, p, target, pred_mask, last_step, node_mask):

    bsize = p.shape[0]
    nnodes = p.shape[1]
    n_steps = torch.max(last_step)

    if pred_mask is not None:
        mask = pred_mask[:,:,:n_steps].unsqueeze(1).transpose(-1,-2).expand_as(p[:,:,:n_steps,:]) \
            + torch.eye(nnodes,
                        device=device
                        ).unsqueeze(0).unsqueeze(2).expand_as(p[:,:,:n_steps, :]).bool()
        p[:,:,:n_steps,:] = torch.where(
            mask,
            p[:,:,:n_steps,:].double(),
            float('-inf')
        ).float()

    p_final = (torch.gather(torch.argmax(p, dim=-1), 2, last_step.repeat(1,nnodes,1))).squeeze()
    loss_pred_final = (p_final == target).long()
    loss_pred_final = node_mask * loss_pred_final
    div = torch.maximum((node_mask).float().sum(dim=1), torch.ones((bsize,), device=device))
    loss_pred_final = torch.sum(loss_pred_final.float(), dim=1)/div
    pred_last_acc = loss_pred_final.sum()
    return pred_last_acc

def dist_test(y, target, node_mask, term_mask, normalise=False):
    loss_dist_steps = fn.mse_loss(y,
                                target,
                                reduction='none'
                                )
    if normalise:
        var_target = torch.var(target, dim=1, unbiased=False, keepdim=True)
        loss_dist_steps = term_mask * node_mask * loss_dist_steps * (1/var_target)
    else:
        loss_dist_steps = term_mask * node_mask * loss_dist_steps
    div = (torch.max((node_mask).float().sum(dim=1),torch.tensor([1.],device=y.device)))
    loss_dist_steps = loss_dist_steps.sum(dim=1)/(div.squeeze(-1))
    loss_dist_steps = loss_dist_steps.sum(dim=1)/(term_mask).sum(dim=-1).squeeze()
    dist_mse = loss_dist_steps.sum()
    return dist_mse

def dist_test_last(y, target, node_mask, last_step, normalise=False):
    nnodes = y.shape[1]
    y_final = (torch.gather(y,2,last_step.repeat(1, nnodes, 1))).squeeze()
    loss_dist_steps = fn.mse_loss(
        y_final,
        target,
        reduction='none'
    )
    if normalise:
        var_target = torch.var(target, dim=1, unbiased=False, keepdim=True)
        loss_dist_steps = node_mask * loss_dist_steps * (1/var_target)
    else:
        loss_dist_steps = node_mask * loss_dist_steps
    div = (torch.max((node_mask).float().sum(dim=1),torch.tensor([1.],device=y.device)))
    loss_dist_steps = loss_dist_steps.sum(dim=1)/(div.squeeze(-1))
    dist_mse = loss_dist_steps.sum()
    return dist_mse


# We assume that either all the nodes in a graph are tested or none
def state_test_mean(y, target, term_mask):
    loss_state_steps = (y==target).float()
    loss_state_steps = term_mask.unsqueeze(1) * loss_state_steps
    loss_state_steps = loss_state_steps.sum(dim=-1)/(term_mask.unsqueeze(1)).sum(dim=-1)
    loss_state_steps = torch.mean(loss_state_steps, dim=1)
    return loss_state_steps.sum()

def state_test_last(y, target, last_step):
    nnodes = y.shape[1]
    y_final = (torch.gather(y,2,last_step.repeat(1, nnodes, 1))).squeeze()
    loss_state_steps = (y_final==target).float()
    loss_state_steps = torch.mean(loss_state_steps, dim=1)
    return loss_state_steps.sum()


## GENERAL UTILITY

# assume a and b have the same shape
def float_or(a, b):
    return a + b - (a * b)
