from abc import ABC, abstractmethod

import torch, pdb
import torch.nn as nn
import torch.nn.functional as fn

import loss_utils as utils

class LossAssembler_():
    def __init__(
            self,
            device,
            logger,
            algos,
            opt
    ):
        self.dev = device
        self.log = logger

        self.algos = algos #now [bf, bfs ] -> [bf] /[bfs]

        self.k = opt['ksamples']

        # compute number of loss terms
        self.nloss_terms = sum([t.nodedim+t.preddim for t in self.algos])

    def train_loss(self, logger, device, model, batch, algo):
        adj, weights, state, pred, term = batch

        # bring all the tensors to the device
        adj = adj.to(device)
        weights = weights.to(device)
        state = state.to(device)
        pred = pred.to(device)
        term = term.to(device)
        
        # k-samples of the trajectory
        k = self.k
        og_bsize = adj.shape[0]
        if abs(self.k) > 1:
            adj_shape = adj.shape
            adj = adj.unsqueeze(1).expand(-1, abs(k), -1, -1).reshape(-1, *adj_shape[1:])
            weights_shape = weights.shape
            weights = weights.unsqueeze(1).expand(-1, abs(k), -1, -1).reshape(-1, *weights_shape[1:])
            state_shape = state.shape
            state = state.unsqueeze(1).expand(
                -1, abs(k),*state_shape[1:]).reshape(-1, *state_shape[1:])
            pred_shape = pred.shape
            pred = pred.unsqueeze(1).expand(-1, abs(k), -1,-1, -1).reshape(-1, *pred_shape[1:])
            term_shape = term.shape
            term = term.unsqueeze(1).expand(-1, abs(k), -1).reshape(-1, *term_shape[1:])

        # batch info + initialisation
        bsize = adj.shape[0]
        nnodes = adj.shape[1]
        max_steps = (nnodes)
        ndim= len(model.ndim) if isinstance(model.ndim, list) else 1
        h = torch.zeros((bsize, nnodes, model.hdim), device=device)
        n_steps = torch.zeros((bsize,), device=device)
        # pos_w = torch.log(torch.mean((1-term).sum(dim=-1)))
        pos_w = torch.mean((1-term).sum(dim=-1))
        batch_loss = [torch.zeros((bsize,1), device=device) for _ in range(self.nloss_terms)]
        batch_term_loss = 0

        # model input
        model_in = state[:,:,:,0].clone()
        #pdb.set_trace()
        for i in range(max_steps):
            y, p, tau, h = model(model_in, h, adj, weights, algo)

            # ensure even if total preddim is 1 that p has 4 dimensions
            if p is not None and p.ndim < 4:
                p = p.unsqueeze(-1)

            s_offset = 0
            p_offset = 0
            mask = (1-term[:,i]).bool().long()

            # termination loss (we assume this is shared)
            batch_term_loss += utils.term_loss(
                tau,
                term[:,i+1],
                pos_w,
                mask
            )

            for t in self.algos:
                # selecting the dimensions for this algo
                y_algo = y[:,:,s_offset:s_offset+t.nodedim].clone()

                # mask computation
                node_mask, pred_mask = t.mask_fn(
                    device,
                    model_in[:,:,s_offset:s_offset+t.nodedim],
                    y_algo,
                    state[:,:,s_offset:s_offset+t.nodedim,i],
                    state[:,:,s_offset:s_offset+t.nodedim,i+1],
                    pred[:,:,p_offset:p_offset+t.preddim,i] if t.preddim > 0 else None
                )
                term_mask = mask if t.tf else mask * term[:,i+1]

                # state loss computation
                batch_loss_state_algo = t.state_loss_fn(
                    device,
                    y_algo,
                    state[:,:,s_offset:s_offset+t.nodedim,i],
                    state[:,:,s_offset:s_offset+t.nodedim,i+1],
                    node_mask,
                    term_mask
                )

                # pred loss computation
                if t.preddim > 0:
                    p_algo = p[:,:,:,p_offset:p_offset+t.preddim]
                    batch_loss_pred_algo = t.pred_loss_fn(
                        device,
                        p_algo,
                        pred[:,:,p_offset:p_offset+t.preddim,i],
                        pred_mask,
                        term_mask
                    )

                # model_in computation
                if t.tf:
                    algo_in = state[:,:, s_offset:s_offset+t.nodedim, i+1]
                else:
                    algo_in = t.transition_fn(
                        device,
                        model_in[:,:,s_offset:s_offset+t.nodedim].clone(),
                        y_algo.clone()
                    )
                model_in[:,:,s_offset:s_offset+t.nodedim] = algo_in.clone()


                l_offset = s_offset+p_offset
                s_offset += t.nodedim
                p_offset += t.preddim

                # reporting the batch loss
                batch_loss[l_offset:l_offset+t.nodedim] = list(
                    map(
                        sum,
                        zip(batch_loss_state_algo,
                            batch_loss[l_offset:l_offset+t.nodedim])
                    )
                )
                l_offset += t.nodedim
                if t.preddim > 0:
                    batch_loss[l_offset:l_offset+t.preddim] = list(
                        map(
                            sum,
                            zip(batch_loss_pred_algo,
                                batch_loss[l_offset:l_offset+t.preddim])
                        )
                    )

                # compute number_steps
                l_offset = s_offset+p_offset
                alg_nloss_terms = t.nodedim+t.preddim
            n_steps += term_mask

        # pick the k-samples
        if abs(self.k) > 1:
            batch_loss = utils.max_of_k_samples(
                list(batch_loss),
                og_bsize,
                k
            )
        # average across batch
        offset = 0
        for t in self.algos:
            for _ in range(t.preddim+t.nodedim):
                if t.tf:
                    batch_loss[offset] = torch.mean(batch_loss[offset]/n_steps)
                else:
                    batch_loss[offset] = torch.mean(batch_loss[offset])
                offset += 1

        loss = []
        offset = 0
        for t in self.algos:
            task_loss = 0
            for _ in range(t.preddim+t.nodedim):
                task_loss += batch_loss[offset]
                offset +=1
            loss.append(task_loss)

        batch_term_loss = torch.mean(batch_term_loss)
        loss.append(batch_term_loss)

        return loss


    @torch.no_grad()
    def val_loss(self, logger, device, model, batch, algo):
        batch_loss = self.train_loss(logger, device, model, batch, algo)

        return batch_loss

    @torch.no_grad()
    def test_loss(self, logger, device, model, batch, algo):
        adj, weights, state, pred, term = batch

        # bring all the tensors to the device
        adj = adj.to(device)
        weights = weights.to(device)
        state = state.to(device)
        pred = pred.to(device)
        term = term.to(device)

        # batch info + initialisation
        bsize = adj.shape[0]
        nnodes = adj.shape[1]
        max_steps = (nnodes)
        ndim= len(model.ndim) if isinstance(model.ndim, list) else 1
        h = torch.zeros((bsize, nnodes, model.hdim), device=device)
        n_steps = 0
        pos_w = torch.mean((1-term).sum(dim=-1))

        # recording outputs
        model_steps = torch.zeros(state.shape, device=device)
        model_p_out = torch.zeros((bsize,nnodes,nnodes,pred.shape[2],term.shape[1]),
                                  device=device) if pred.ndim > 2 else None
        model_tau = torch.zeros(term.shape, device=device)
        pred_mask = torch.zeros(pred.shape, device=device) if pred.ndim > 2 else None
        node_mask = torch.zeros(state.shape, device=device)

        # model input
        model_in = state[:,:,:,0]

        # running the model
        for i in range(max_steps):
            y, p, tau, h = model(model_in, h, adj, weights, algo)

            # ensure even if total preddim is 1 that p has 4 dimensions
            if p is not None and p.ndim < 4:
                p = p.unsqueeze(-1)

            s_offset = 0
            p_offset = 0
            mask = (1-term[:,i]).bool().long()
            n_steps += 1

            # recording tau
            model_tau[:,i] = tau.squeeze()

            for t in self.algos:
                # selecting the dimensions for this algo
                y_algo = y[:,:,s_offset:s_offset+t.nodedim]

                # compute the masks
                step_node_mask, step_pred_mask = t.mask_fn(
                    device,
                    model_in[:,:,s_offset:s_offset+t.nodedim],
                    y_algo,
                    state[:,:,s_offset:s_offset+t.nodedim,i],
                    state[:,:,s_offset:s_offset+t.nodedim,i+1],
                    pred[:,:,p_offset:p_offset+t.preddim,i] if t.preddim > 0 else None
                )
                node_mask[:,:,s_offset:s_offset+t.nodedim,i] = step_node_mask

                if t.preddim > 0:
                    pred_mask[:,:,p_offset:p_offset+t.preddim,i] = step_pred_mask

                # next state
                algo_in = t.eval_transition_fn(
                    device,
                    model_in[:,:,s_offset:s_offset+t.nodedim],
                    y_algo
                )
                # import pdb; pdb.set_trace()
                model_in[:, :, s_offset:s_offset+t.nodedim] = algo_in
                model_steps[:,:, s_offset:s_offset+t.nodedim, i] = algo_in
                if t.preddim > 0:
                    p_algo = p[:,:,:,p_offset:p_offset+t.preddim]
                    model_p_out[:,:,:,p_offset:p_offset+t.preddim, i] = p_algo

                # compute number_steps
                l_offset = s_offset+p_offset
                alg_nloss_terms = t.nodedim+t.preddim

                # increase offset
                s_offset += t.nodedim
                p_offset += t.preddim


        tf = False
        for t in self.algos:
            tf = tf or t.tf

        if not tf:
            model_tau = term

        # sequence length calculations
        last_step = utils.get_laststep(model_tau[:,:max_steps].unsqueeze(1))
        true_last_step = torch.sum(1-term[:,1:max_steps+1],dim=-1).long().squeeze()
        term_mask = (torch.cumsum(
            torch.cumsum(
                torch.sigmoid(
                    model_tau[:,:max_steps]
                )>0.5,
                dim=-1
            ),
            dim=-1
        ) <= 1).long().squeeze()
        term_mask = 1-term[:,:max_steps]

        # collecting test score
        test_acc_list = []

        # evaluate model
        s_offset = 0
        p_offset = 0
        for t in self.algos:
            if t.tf:
                state_mean_acc = t.state_mean_acc_fn(
                    device,
                    model_steps[:,:, s_offset:s_offset+t.nodedim,:n_steps],
                    state[:,:, s_offset:s_offset+t.nodedim,1:n_steps+1],
                    node_mask[:,:, s_offset:s_offset+t.nodedim,:n_steps],
                    term_mask
                )
                test_acc_list += state_mean_acc
            state_last_acc = t.state_last_acc_fn(
                device,
                model_steps[:,:, s_offset:s_offset+t.nodedim,:n_steps],
                state[:,:, s_offset:s_offset+t.nodedim, 1:n_steps+1],
                node_mask[:,:, s_offset:s_offset+t.nodedim,:n_steps],
                last_step #  if t.tf else true_last_step
            )
            test_acc_list += state_last_acc
            if t.preddim > 0:
                if t.tf:
                    pred_mean_acc = t.pred_mean_acc_fn(
                        device,
                        model_p_out[:,:,:,p_offset:p_offset+t.preddim,:n_steps],
                        pred[:,:,p_offset:p_offset+t.preddim,:n_steps],
                        pred_mask[:,:,p_offset:p_offset+t.preddim,:n_steps],
                        term_mask
                    )
                    test_acc_list += pred_mean_acc
                pred_last_acc = t.pred_last_acc_fn(
                    device,
                    model_p_out[:,:,:,p_offset:p_offset+t.preddim,:n_steps],
                    pred[:,:, p_offset:p_offset+t.preddim, :n_steps],
                    pred_mask[:,:, p_offset:p_offset+t.preddim,:n_steps],
                    last_step  # if t.tf else true_last_step
                )
                test_acc_list += pred_last_acc
            # new offsets
            s_offset += t.nodedim
            p_offset += t.preddim
            if t.tf:
                term_acc = 1-(torch.abs(last_step-utils.get_laststep(term.unsqueeze(1)))/utils.get_laststep(term.unsqueeze(1)))
                test_acc_list.append(term_acc.sum())

        return test_acc_list

### Loss class to construct the loss function for any arbitrary list of algos
class LossAssembler():
    def __init__(
            self,
            device,
            logger,
            algos,
            opt
    ):
        self.dev = device
        self.log = logger

        self.algos = algos #now [bf, bfs ] -> [bf] /[bfs]

        self.k = opt['ksamples']

        # compute number of loss terms
        self.nloss_terms = sum([t.nodedim+t.preddim for t in self.algos])

    def train_loss(self, logger, device, model, batch):
        adj, weights, state, pred, term = batch

        # bring all the tensors to the device
        adj = adj.to(device)
        weights = weights.to(device)
        state = state.to(device)
        pred = pred.to(device)
        term = term.to(device)
        
        # k-samples of the trajectory
        k = self.k
        og_bsize = adj.shape[0]
        if abs(self.k) > 1:
            adj_shape = adj.shape
            adj = adj.unsqueeze(1).expand(-1, abs(k), -1, -1).reshape(-1, *adj_shape[1:])
            weights_shape = weights.shape
            weights = weights.unsqueeze(1).expand(-1, abs(k), -1, -1).reshape(-1, *weights_shape[1:])
            state_shape = state.shape
            state = state.unsqueeze(1).expand(
                -1, abs(k),*state_shape[1:]).reshape(-1, *state_shape[1:])
            pred_shape = pred.shape
            pred = pred.unsqueeze(1).expand(-1, abs(k), -1,-1, -1).reshape(-1, *pred_shape[1:])
            term_shape = term.shape
            term = term.unsqueeze(1).expand(-1, abs(k), -1).reshape(-1, *term_shape[1:])

        # batch info + initialisation
        bsize = adj.shape[0]
        nnodes = adj.shape[1]
        max_steps = (nnodes)
        ndim= len(model.ndim) if isinstance(model.ndim, list) else 1
        h = torch.zeros((bsize, nnodes, model.hdim*ndim), device=device)
        n_steps = torch.zeros((bsize,), device=device)
        # pos_w = torch.log(torch.mean((1-term).sum(dim=-1)))
        pos_w = torch.mean((1-term).sum(dim=-1))
        batch_loss = [torch.zeros((bsize,1), device=device) for _ in range(self.nloss_terms)]
        batch_term_loss = 0

        # model input
        model_in = state[:,:,:,0].clone()

        for i in range(max_steps):
            y, p, tau, h = model(model_in, h, adj, weights)

            # ensure even if total preddim is 1 that p has 4 dimensions
            if p is not None and p.ndim < 4:
                p = p.unsqueeze(-1)

            s_offset = 0
            p_offset = 0
            mask = (1-term[:,i]).bool().long()

            # termination loss (we assume this is shared)
            batch_term_loss += utils.term_loss(
                tau,
                term[:,i+1],
                pos_w,
                mask
            )

            for t in self.algos:
                # selecting the dimensions for this algo
                y_algo = y[:,:,s_offset:s_offset+t.nodedim].clone()

                # mask computation
                node_mask, pred_mask = t.mask_fn(
                    device,
                    model_in[:,:,s_offset:s_offset+t.nodedim],
                    y_algo,
                    state[:,:,s_offset:s_offset+t.nodedim,i],
                    state[:,:,s_offset:s_offset+t.nodedim,i+1],
                    pred[:,:,p_offset:p_offset+t.preddim,i] if t.preddim > 0 else None
                )
                term_mask = mask if t.tf else mask * term[:,i+1]

                # state loss computation
                batch_loss_state_algo = t.state_loss_fn(
                    device,
                    y_algo,
                    state[:,:,s_offset:s_offset+t.nodedim,i],
                    state[:,:,s_offset:s_offset+t.nodedim,i+1],
                    node_mask,
                    term_mask
                )

                # pred loss computation
                if t.preddim > 0:
                    p_algo = p[:,:,:,p_offset:p_offset+t.preddim]
                    batch_loss_pred_algo = t.pred_loss_fn(
                        device,
                        p_algo,
                        pred[:,:,p_offset:p_offset+t.preddim,i],
                        pred_mask,
                        term_mask
                    )

                # model_in computation
                if t.tf:
                    algo_in = state[:,:, s_offset:s_offset+t.nodedim, i+1]
                else:
                    algo_in = t.transition_fn(
                        device,
                        model_in[:,:,s_offset:s_offset+t.nodedim].clone(),
                        y_algo.clone()
                    )
                model_in[:,:,s_offset:s_offset+t.nodedim] = algo_in.clone()


                l_offset = s_offset+p_offset
                s_offset += t.nodedim
                p_offset += t.preddim

                # reporting the batch loss
                batch_loss[l_offset:l_offset+t.nodedim] = list(
                    map(
                        sum,
                        zip(batch_loss_state_algo,
                            batch_loss[l_offset:l_offset+t.nodedim])
                    )
                )
                l_offset += t.nodedim
                if t.preddim > 0:
                    batch_loss[l_offset:l_offset+t.preddim] = list(
                        map(
                            sum,
                            zip(batch_loss_pred_algo,
                                batch_loss[l_offset:l_offset+t.preddim])
                        )
                    )

                # compute number_steps
                l_offset = s_offset+p_offset
                alg_nloss_terms = t.nodedim+t.preddim
            n_steps += term_mask

        # pick the k-samples
        if abs(self.k) > 1:
            batch_loss = utils.max_of_k_samples(
                list(batch_loss),
                og_bsize,
                k
            )
        # average across batch
        offset = 0
        for t in self.algos:
            for _ in range(t.preddim+t.nodedim):
                if t.tf:
                    batch_loss[offset] = torch.mean(batch_loss[offset]/n_steps)
                else:
                    batch_loss[offset] = torch.mean(batch_loss[offset])
                offset += 1

        loss = []
        offset = 0
        for t in self.algos:
            task_loss = 0
            for _ in range(t.preddim+t.nodedim):
                task_loss += batch_loss[offset]
                offset +=1
            loss.append(task_loss)

        batch_term_loss = torch.mean(batch_term_loss)
        loss.append(batch_term_loss)

        return loss


    @torch.no_grad()
    def val_loss(self, logger, device, model, batch):
        batch_loss = self.train_loss(logger, device, model, batch)
        # val_loss = []
        # offset = 0
        # for t in self.algos:
        #     for _ in range(t.nodedim):
        #         offset += 1
        #     for _ in range(t.preddim):
        #         val_loss.append(batch_loss[offset])
        #         offset += 1

        # if len(val_loss) == 0:
        #     val_loss = batch_loss

        return batch_loss

    @torch.no_grad()
    def test_loss(self, logger, device, model, batch):
        adj, weights, state, pred, term = batch

        # bring all the tensors to the device
        adj = adj.to(device)
        weights = weights.to(device)
        state = state.to(device)
        pred = pred.to(device)
        term = term.to(device)

        # batch info + initialisation
        bsize = adj.shape[0]
        nnodes = adj.shape[1]
        max_steps = (nnodes)
        ndim= len(model.ndim) if isinstance(model.ndim, list) else 1
        h = torch.zeros((bsize, nnodes, model.hdim*ndim), device=device)
        n_steps = 0
        pos_w = torch.mean((1-term).sum(dim=-1))

        # recording outputs
        model_steps = torch.zeros(state.shape, device=device)
        model_p_out = torch.zeros((bsize,nnodes,nnodes,pred.shape[2],term.shape[1]),
                                  device=device) if pred.ndim > 2 else None
        model_tau = torch.zeros(term.shape, device=device)
        pred_mask = torch.zeros(pred.shape, device=device) if pred.ndim > 2 else None
        node_mask = torch.zeros(state.shape, device=device)

        # model input
        model_in = state[:,:,:,0]

        # running the model
        for i in range(max_steps):
            y, p, tau, h = model(model_in, h, adj, weights)

            # ensure even if total preddim is 1 that p has 4 dimensions
            if p is not None and p.ndim < 4:
                p = p.unsqueeze(-1)

            s_offset = 0
            p_offset = 0
            mask = (1-term[:,i]).bool().long()
            n_steps += 1

            # recording tau
            model_tau[:,i] = tau.squeeze()

            for t in self.algos:
                # selecting the dimensions for this algo
                y_algo = y[:,:,s_offset:s_offset+t.nodedim]

                # compute the masks
                step_node_mask, step_pred_mask = t.mask_fn(
                    device,
                    model_in[:,:,s_offset:s_offset+t.nodedim],
                    y_algo,
                    state[:,:,s_offset:s_offset+t.nodedim,i],
                    state[:,:,s_offset:s_offset+t.nodedim,i+1],
                    pred[:,:,p_offset:p_offset+t.preddim,i] if t.preddim > 0 else None
                )
                node_mask[:,:,s_offset:s_offset+t.nodedim,i] = step_node_mask

                if t.preddim > 0:
                    pred_mask[:,:,p_offset:p_offset+t.preddim,i] = step_pred_mask

                # next state
                algo_in = t.eval_transition_fn(
                    device,
                    model_in[:,:,s_offset:s_offset+t.nodedim],
                    y_algo
                )
                # import pdb; pdb.set_trace()
                model_in[:, :, s_offset:s_offset+t.nodedim] = algo_in
                model_steps[:,:, s_offset:s_offset+t.nodedim, i] = algo_in
                if t.preddim > 0:
                    p_algo = p[:,:,:,p_offset:p_offset+t.preddim]
                    model_p_out[:,:,:,p_offset:p_offset+t.preddim, i] = p_algo

                # compute number_steps
                l_offset = s_offset+p_offset
                alg_nloss_terms = t.nodedim+t.preddim

                # increase offset
                s_offset += t.nodedim
                p_offset += t.preddim


        tf = False
        for t in self.algos:
            tf = tf or t.tf

        if not tf:
            model_tau = term

        # sequence length calculations
        last_step = utils.get_laststep(model_tau[:,:max_steps].unsqueeze(1))
        true_last_step = torch.sum(1-term[:,1:max_steps+1],dim=-1).long().squeeze()
        term_mask = (torch.cumsum(
            torch.cumsum(
                torch.sigmoid(
                    model_tau[:,:max_steps]
                )>0.5,
                dim=-1
            ),
            dim=-1
        ) <= 1).long().squeeze()
        term_mask = 1-term[:,:max_steps]

        # collecting test score
        test_acc_list = []

        # evaluate model
        s_offset = 0
        p_offset = 0
        for t in self.algos:
            if t.tf:
                state_mean_acc = t.state_mean_acc_fn(
                    device,
                    model_steps[:,:, s_offset:s_offset+t.nodedim,:n_steps],
                    state[:,:, s_offset:s_offset+t.nodedim,1:n_steps+1],
                    node_mask[:,:, s_offset:s_offset+t.nodedim,:n_steps],
                    term_mask
                )
                test_acc_list += state_mean_acc
            state_last_acc = t.state_last_acc_fn(
                device,
                model_steps[:,:, s_offset:s_offset+t.nodedim,:n_steps],
                state[:,:, s_offset:s_offset+t.nodedim, 1:n_steps+1],
                node_mask[:,:, s_offset:s_offset+t.nodedim,:n_steps],
                last_step #  if t.tf else true_last_step
            )
            test_acc_list += state_last_acc
            if t.preddim > 0:
                if t.tf:
                    pred_mean_acc = t.pred_mean_acc_fn(
                        device,
                        model_p_out[:,:,:,p_offset:p_offset+t.preddim,:n_steps],
                        pred[:,:,p_offset:p_offset+t.preddim,:n_steps],
                        pred_mask[:,:,p_offset:p_offset+t.preddim,:n_steps],
                        term_mask
                    )
                    test_acc_list += pred_mean_acc
                pred_last_acc = t.pred_last_acc_fn(
                    device,
                    model_p_out[:,:,:,p_offset:p_offset+t.preddim,:n_steps],
                    pred[:,:, p_offset:p_offset+t.preddim, :n_steps],
                    pred_mask[:,:, p_offset:p_offset+t.preddim,:n_steps],
                    last_step  # if t.tf else true_last_step
                )
                test_acc_list += pred_last_acc
            # new offsets
            s_offset += t.nodedim
            p_offset += t.preddim
            if t.tf:
                term_acc = 1-(torch.abs(last_step-utils.get_laststep(term.unsqueeze(1)))/utils.get_laststep(term.unsqueeze(1)))
                test_acc_list.append(term_acc.sum())

        return test_acc_list


## loss creater function
def create_loss_class(task_name, args):
    task_split = task_name.split('_')
    opt = {}
    if len(task_split) > 1:
        opt['tf'] = False
    else:
        opt['tf'] = True

    algo = task_split[-1]
    if algo == 'bfs':
        return BFSLoss(opt)
    elif algo == 'bf':
        return BellmanFordLoss(opt)
    elif algo == 'prims':
        if args['hidekeys'] and opt['tf']:
            return SequentialPriorityQueueHiddenKeys(opt)
        else:
            return SequentialPriorityQueueLoss(opt)
    elif algo == 'dijkstra':
        if args['hidekeys'] and opt['tf']:
            return SequentialPriorityQueueHiddenKeys(opt)
        else:
            return SequentialPriorityQueueLoss(opt)
    elif algo == 'mostrelseq':
        if args['hidekeys'] and opt['tf']:
            return SequentialPriorityQueueHiddenKeys(opt)
        else:
            return SequentialPriorityQueueLoss(opt)
    elif algo == 'dfs':
        if args['hidekeys'] and opt['tf']:
            return SequentialPriorityQueueHiddenKeys(opt)
        else:
            return SequentialPriorityQueueLoss(opt)
    elif algo == 'widest':
        if args['hidekeys'] and opt['tf']:
            return SequentialPriorityQueueHiddenKeys(opt)
        else:
            return SequentialPriorityQueueLoss(opt)
    elif algo == 'mostrelpar':
        return BellmanFordLoss(opt)
    elif algo == 'widestpar':
        return BellmanFordLoss(opt)
    else:
        raise NotImplementedError

## Abstract loss
class AbstractLoss(ABC):
    """
    Abstract class that defines the functions each loss clas must implement
    """

    # functions that must be implemented
    @abstractmethod
    def state_loss_fn(self, device, state_out, state_in, state_true, node_mask, term_mask):
        pass

    @abstractmethod
    def pred_loss_fn(self, device, pred_out, pred_true, pred_mask, term_mask):
        pass

    @abstractmethod
    def mask_fn(self, device, state_in, state_out, state_true_in, state_true, pred_out):
        pass

    @abstractmethod
    def transition_fn(self, device, state_in, state_out):
        pass

    @abstractmethod
    def eval_transition_fn(self, device, state_in, state_out):
        pass

    @abstractmethod
    def state_last_acc_fn(self, device, state_out, state_true, node_mask, last_step):
        pass

    @abstractmethod
    def pred_last_acc_fn(self, device, pred_out, pred_true, pred_mask, last_step):
        pass

    @abstractmethod
    def state_mean_acc_fn(self, device, state_out, state_true, node_mask, term_mask):
        pass

    @abstractmethod
    def pred_mean_acc_fn(self, device, pred_out, pred_true, pred_mask, term_mask):
        pass

    # variables that must be instantiated
    @property
    @abstractmethod
    def tf(self):
        pass

    @property
    @abstractmethod
    def nodedim(self):
        pass

    @property
    @abstractmethod
    def preddim(self):
        pass

## BFS loss class
class BFSLoss(AbstractLoss):
    tf = True
    nodedim = 1
    preddim = 0

    def __init__(self, opt):
        self.tf = opt['tf']

    def state_loss_fn(self, device, state_out, state_in, state_true, node_mask, term_mask):
        return utils.state_loss(state_out,
                                state_true,
                                term_mask.unsqueeze(1)
                                )

    def pred_loss_fn(self, device, pred_out, pred_true, pred_mask, term_mask):
        return 0

    def mask_fn(self, device, state_in, state_out, state_true_in, state_true, pred_true):
        return torch.ones(state_in.shape, device=device), None

    def transition_fn(self, device, state_in, state_out):
        return (torch.sigmoid(state_out)>=0.5).float()

    def eval_transition_fn(self, device, state_in, state_out):
        return (torch.sigmoid(state_out)>=0.5).float()

    def state_mean_acc_fn(self, device, model_state, state_true, node_mask, term_mask):
        state_mean_acc = utils.state_test_mean(
            model_state.squeeze(2),
            state_true.squeeze(2),
            term_mask
        )
        return [state_mean_acc]

    def state_last_acc_fn(self, device, model_state, state_true, node_mask, last_step):
        state_last_acc = utils.state_test_last(
            model_state.squeeze(2),
            state_true.squeeze(2)[:,:,-1],
            last_step.unsqueeze(1).unsqueeze(1)
        )
        return [state_last_acc]

    def pred_last_acc_fn(self, device, pred_out, pred_true, pred_mask, last_step):
        # no need to implement this, it should never be called
        pass

    def pred_mean_acc_fn(self, device, pred_out, pred_true, pred_mask, term_mask):
        # no need to implement this, it should never be called
        pass

## Bellman-Ford loss class
class BellmanFordLoss(AbstractLoss):
    nodedim = 1
    preddim = 1
    tf = True
    def __init__(self, opt):
        self.tf = opt['tf']

    def process_pred_true(self, device, pred_true, pred_mask):
        pred_true = torch.where(
            pred_mask.bool(),
            pred_true,
            torch.arange(
                0,
                pred_true.shape[1],
                device = device
            ).unsqueeze(0).expand_as(pred_true).float(),
        )
        return pred_true

    def state_loss_fn(self, device, state_out, state_in, state_true, node_mask, term_mask):
        return utils.dist_loss(
            state_out.squeeze(),
            state_true.squeeze(),
            node_mask.squeeze(),
            term_mask.unsqueeze(1)
        )

    def pred_loss_fn(self, device, pred_out, pred_true, pred_mask, term_mask):
        pred_true = self.process_pred_true(
            device,
            pred_true.squeeze(2),
            pred_mask.squeeze(2)
        )
        return utils.pred_loss(
            device,
            pred_out,
            pred_true,
            None,
            term_mask,
            pred_mask.squeeze(2)
        )

    def mask_fn(self, device, state_in, state_out, state_true_in, state_true, pred_true):
        mask = (~(pred_true.float() == float('inf'))).float()
        return mask, mask

    def transition_fn(self, device, state_in, state_out):
        return state_out

    def eval_transition_fn(self, device, state_in, state_out):
        return state_out

    def state_mean_acc_fn(self, device, model_state, state_true, node_mask, term_mask):
        state_mean_acc = utils.dist_test(
            model_state.squeeze(2),
            state_true.squeeze(2),
            node_mask.squeeze(2),
            term_mask.unsqueeze(1)
        )
        return [state_mean_acc]

    def state_last_acc_fn(self, device, model_state, state_true, node_mask, last_step):
        state_last_acc = utils.dist_test_last(
            model_state.squeeze(2),
            state_true.squeeze(2)[:,:,-1],
            node_mask.squeeze(2)[:,:,-1],
            last_step.unsqueeze(1).unsqueeze(1)
        )
        return [state_last_acc]

    def pred_last_acc_fn(self, device, pred_out, pred_true, pred_mask, last_step):
        pred_true = self.process_pred_true(
            device,
            pred_true.squeeze(2)[:,:,-1],
            pred_mask.squeeze(2)[:,:,-1]
        )
        pred_last_acc = utils.pred_test_last(
            device,
            pred_out.squeeze(3).transpose(-1,-2),
            pred_true,
            None,
            last_step.unsqueeze(1).unsqueeze(1),
            pred_mask.squeeze(2)[:,:,-1] # todo add this to the loss_utils
        )
        return [pred_last_acc]

    def pred_mean_acc_fn(self, device, pred_out, pred_true, pred_mask, term_mask):
        pred_true = self.process_pred_true(
            device,
            pred_true.squeeze(2),
            pred_mask.squeeze(2)
        )
        pred_mean_acc = utils.pred_test_mean(
            device,
            pred_out.squeeze(3).transpose(-1,-2),
            pred_true.squeeze(2),
            None,
            term_mask,
            pred_mask.squeeze(2)
        )
        return [pred_mean_acc]


## Prims loss class
class SequentialPriorityQueueHiddenKeys(AbstractLoss):
    nodedim = 2
    preddim = 1
    tf = True
    def __init__(self, opt):
        self.tf = opt['tf']

    def process_pred_true(self, device, pred_true, pred_mask):
        filler = torch.arange(
                0,
                pred_true.shape[1],
                device = device
            ).unsqueeze(0)
        if pred_true.ndim > 2:
            filler = filler.unsqueeze(2)
        pred_true = torch.where(
            pred_mask.bool(),
            pred_true.float(),
            filler.expand_as(pred_true).float(),
        )
        return pred_true

    def last_step_to_term_mask(self, device, last_step, n_steps):
        b_size = last_step.shape[0]
        step_num = torch.arange(0,n_steps, device=device).unsqueeze(0).expand(b_size,
                                                                              n_steps)
        zeros = torch.zeros((b_size,n_steps),device=device)
        ones = torch.ones((b_size,n_steps),device=device)
        return torch.where(step_num <= last_step.unsqueeze(1),
                           ones,
                           zeros
                           )


    def state_loss_fn(self, device, state_out, state_in, state_true, node_mask, term_mask):

        if self.tf:
            prim_next_node = torch.argmax(state_true[:,:,0]-state_in[:,:,0], dim=1)
            prim_next_node = torch.where(
                (state_in[:,:,0] == state_true[:,:,0]).all(1),
                torch.argmax(state_in[:,:,0].long(), dim=1),
                prim_next_node
            ).squeeze()
            # masking out nodes already choosen
            state_out[:,:,0] = torch.where(
                state_in[:,:,0].bool(),
                torch.tensor([float('-inf')],device=device).float(),
                state_out[:,:,0]
            )
            state_loss = utils.next_node_loss(state_out[:,:,0], prim_next_node, term_mask)
        else:
            state_loss = utils.state_loss(state_out[:,:,0], state_true[:,:,0], term_mask)

        # computing the loss for the key
        key_loss = utils.l1_dist_loss(
            state_out[:,:,1],
            state_true[:,:,1],
            node_mask.squeeze(),
            term_mask.unsqueeze(1)
        )
        return [state_loss, key_loss]

    def pred_loss_fn(self, device, pred_out, pred_true, pred_mask, term_mask):
        pred_pred_mask = (pred_true != float('inf')).long()
        pred_mask = pred_mask.long()
        pred_true = self.process_pred_true(
            device,
            pred_true.squeeze(2),
            pred_pred_mask.squeeze(2)
        )
        return [ utils.pred_loss(
            device,
            pred_out.squeeze(),
            pred_true.squeeze(),
            pred_pred_mask.squeeze(2).bool(),
            term_mask,
            pred_mask.squeeze(2)
        ) ]

    def mask_fn(self, device, state_in, state_out, state_true_in, state_true, pred_true):
        node_mask = (state_true[:,:,0]-state_true_in[:,:,0]).long().unsqueeze(-1)
        return node_mask, node_mask

    def transition_fn(self, device, state_in, state_out):
        # ensuring there are no negative predictions because they break masking
        state_out_pos = state_out[:,:,0] \
            - torch.min(state_out[:,:,0], dim=-1)[0].unsqueeze(1)\
            + 1e-3
        # masking already chosen nodes
        state_masked = (1-state_in[:,:,0]) * state_out_pos \
            + state_in[:,:,0] * torch.tensor([[1e-10]], device=device).float()
        new_state = state_in[:,:,0]+fn.gumbel_softmax(state_masked,tau=1,hard=True,dim=1)
        new_key = (new_state-state_in[:,:,0]) * state_out[:,:,1] \
            + (1-(new_state-state_in[:,:,0])) * state_in[:,:,1]
        return torch.stack([new_state, new_key], dim=-1)

    def eval_transition_fn(self, device, state_in, state_out):
        # ensuring there are no negative predictions because they break masking
        state_out[:,:,0] = state_out[:,:,0] \
            - torch.min(state_out[:,:,0], dim=-1)[0].unsqueeze(1)\
            + 1e-3
        # masking already chosen nodes
        state_masked = (1-state_in[:,:,0]) * state_out[:,:,0].clone() \
            + state_in[:,:,0] * torch.tensor([[1e-10]], device=device).float()
        state_idx = torch.argmax(state_masked,keepdim=True, dim=1)
        new_state = torch.scatter(state_in[:,:,0], 1, state_idx, 1)
        new_key = (new_state-state_in[:,:,0]) * state_out[:,:,1] \
            + (1-(new_state-state_in[:,:,0])) * state_in[:,:,1]
        return torch.stack([new_state, new_key], dim=-1)

    def state_mean_acc_fn(self, device, model_state, state_true, node_mask, term_mask):
        n_steps = term_mask.shape[1]
        next_node_model = torch.argmax(model_state[:,:,0].clone(), dim=1).long()
        next_node_model[:,1:n_steps] = torch.argmax(
            model_state[:,:,0,1:n_steps]\
            -model_state[:,:,0,0:n_steps-1],
            dim=1
        )
        next_node_true = torch.argmax(state_true[:,:,0].clone(), dim=1).long()
        next_node_true[:,1:n_steps] = torch.argmax(
            state_true[:,:,0,1:n_steps]\
            -state_true[:,:,0,0:n_steps-1],
            dim=1
        )
        state_mean_acc = utils.next_node_test(
            next_node_model,
            next_node_true,
            term_mask
        )
        key_mean_acc = utils.dist_test(
            model_state[:,:,1],
            state_true[:,:,1],
            node_mask[:,:,1],
            term_mask.unsqueeze(1)
        )
        return [state_mean_acc, key_mean_acc]

    def state_last_acc_fn(self, device, model_state, state_true, node_mask, last_step):
        state_last_acc = utils.state_test_last(
            model_state[:,:,0],
            state_true[:,:,0,-1],
            last_step.unsqueeze(1).unsqueeze(1)
        )
        term_mask = self.last_step_to_term_mask(device,last_step,state_true.shape[-1])
        key_final_acc = utils.dist_test(
            model_state[:,:,1],
            state_true[:,:,1],
            node_mask[:,:,1],
            term_mask.unsqueeze(1)
        )
        return [state_last_acc, key_final_acc]

    def pred_last_acc_fn(self, device, pred_out, pred_true, pred_mask, last_step):
        pred_pred_mask = (pred_true != float('inf')).long()
        pred_mask = pred_mask.long()
        pred_true = self.process_pred_true(
            device,
            pred_true.squeeze(2),
            pred_mask.squeeze(2)
        )
        term_mask = self.last_step_to_term_mask(device,last_step,pred_out.shape[-1])
        pred_final_acc = utils.pred_test_mean(
            device,
            pred_out.squeeze(3).transpose(-1,-2),
            pred_true.squeeze(2),
            pred_pred_mask.squeeze(2).bool(),
            term_mask,
            pred_mask.squeeze(2)
        )
        return [pred_final_acc]

    def pred_mean_acc_fn(self, device, pred_out, pred_true, pred_mask, term_mask):
        pred_pred_mask = (pred_true != float('inf')).long()
        pred_mask = pred_mask.long()
        pred_true = self.process_pred_true(
            device,
            pred_true.squeeze(2),
            pred_mask.squeeze(2)
        )
        pred_mean_acc = utils.pred_test_mean(
            device,
            pred_out.squeeze(3).transpose(-1,-2),
            pred_true.squeeze(2),
            pred_pred_mask.squeeze(2).bool(),
            term_mask,
            pred_mask.squeeze(2)
        )
        return [pred_mean_acc]

class SequentialPriorityQueueLoss(AbstractLoss):
    nodedim = 2
    preddim = 1
    tf = True
    def __init__(self, opt):
        self.tf = opt['tf']

    def process_pred_true(self, device, pred_true, pred_mask):
        filler = torch.arange(
                0,
                pred_true.shape[1],
                device = device
            ).unsqueeze(0)
        if pred_true.ndim > 2:
            filler = filler.unsqueeze(2)
        pred_true = torch.where(
            pred_mask.bool(),
            pred_true.float(),
            filler.expand_as(pred_true).float(),
        )
        return pred_true

    def state_loss_fn(self, device, state_out, state_in, state_true, node_mask, term_mask):

        if self.tf:
            prim_next_node = torch.argmax(state_true[:,:,0]-state_in[:,:,0], dim=1)
            prim_next_node = torch.where(
                (state_in[:,:,0] == state_true[:,:,0]).all(1),
                torch.argmax(state_in[:,:,0].long(), dim=1),
                prim_next_node
            ).squeeze()
            # masking out nodes already choosen
            state_out[:,:,0] = torch.where(
                state_in[:,:,0].bool(),
                torch.tensor([float('-inf')],device=device).float(),
                state_out[:,:,0]
            )
            state_loss = utils.next_node_loss(state_out[:,:,0], prim_next_node, term_mask)
        else:
            state_loss = utils.state_loss(state_out[:,:,0], state_true[:,:,0], term_mask)

        # computing the loss for the key
        key_loss = utils.l1_dist_loss(
            state_out[:,:,1],
            state_true[:,:,1],
            node_mask.squeeze(),
            term_mask.unsqueeze(1)
        )
        return [state_loss, key_loss]

    def pred_loss_fn(self, device, pred_out, pred_true, pred_mask, term_mask):
        pred_pred_mask = (pred_true != float('inf')).long()
        pred_mask = pred_mask.long()
        pred_true = self.process_pred_true(
            device,
            pred_true.squeeze(2),
            pred_pred_mask.squeeze(2)
        )
        return [ utils.pred_loss(
            device,
            pred_out.squeeze(),
            pred_true.squeeze(),
            pred_pred_mask.squeeze(2).bool(),
            term_mask,
            pred_mask.squeeze(2)
        ) ]

    def mask_fn(self, device, state_in, state_out, state_true_in, state_true, pred_true):
        node_mask = (~(pred_true.float() == float('inf'))).float()
        pred_mask = (state_true_in[:,:,1] != state_true[:,:,1]).float().unsqueeze(2)
        return node_mask, node_mask

    def transition_fn(self, device, state_in, state_out):
        # ensuring there are no negative predictions because they break masking
        state_out_pos = state_out[:,:,0] \
            - torch.min(state_out[:,:,0], dim=-1)[0].unsqueeze(1)\
            + 1e-3
        # masking already chosen nodes
        state_masked = (1-state_in[:,:,0]) * state_out_pos \
            + state_in[:,:,0] * torch.tensor([[1e-10]], device=device).float()
        new_state = state_in[:,:,0]+fn.gumbel_softmax(state_masked,tau=1,hard=True,dim=1)
        new_key = state_out[:,:,1]
        return torch.stack([new_state, new_key], dim=-1)

    def eval_transition_fn(self, device, state_in, state_out):
        # ensuring there are no negative predictions because they break masking
        state_out[:,:,0] = state_out[:,:,0] \
            - torch.min(state_out[:,:,0], dim=-1)[0].unsqueeze(1)\
            + 1e-3
        # masking already chosen nodes
        state_masked = (1-state_in[:,:,0]) * state_out[:,:,0].clone() \
            + state_in[:,:,0] * torch.tensor([[1e-10]], device=device).float()
        state_idx = torch.argmax(state_masked,keepdim=True, dim=1)
        new_state = torch.scatter(state_in[:,:,0], 1, state_idx, 1)
        new_key = state_out[:,:,1]
        return torch.stack([new_state, new_key], dim=-1)

    def state_mean_acc_fn(self, device, model_state, state_true, node_mask, term_mask):
        n_steps = term_mask.shape[1]
        next_node_model = torch.argmax(model_state[:,:,0].clone(), dim=1).long()
        next_node_model[:,1:n_steps] = torch.argmax(
            model_state[:,:,0,1:n_steps]\
            -model_state[:,:,0,0:n_steps-1],
            dim=1
        )
        next_node_true = torch.argmax(state_true[:,:,0].clone(), dim=1).long()
        next_node_true[:,1:n_steps] = torch.argmax(
            state_true[:,:,0,1:n_steps]\
            -state_true[:,:,0,0:n_steps-1],
            dim=1
        )
        state_mean_acc = utils.next_node_test(
            next_node_model,
            next_node_true,
            term_mask
        )
        key_mean_acc = utils.dist_test(
            model_state[:,:,1],
            state_true[:,:,1],
            node_mask[:,:,1],
            term_mask.unsqueeze(1)
        )
        return [state_mean_acc, key_mean_acc]

    def state_last_acc_fn(self, device, model_state, state_true, node_mask, last_step):
        state_last_acc = utils.state_test_last(
            model_state[:,:,0],
            state_true[:,:,0,-1],
            last_step.unsqueeze(1).unsqueeze(1)
        )
        key_last_acc = utils.dist_test_last(
            model_state[:,:,1],
            state_true[:,:,1,-1],
            node_mask[:,:,1,-1],
            last_step.unsqueeze(1).unsqueeze(1)
        )
        return [state_last_acc, key_last_acc]

    def pred_last_acc_fn(self, device, pred_out, pred_true, pred_mask, last_step):
        pred_pred_mask = (pred_true != float('inf')).long()
        pred_mask = pred_mask.long()
        pred_true = self.process_pred_true(
            device,
            pred_true.squeeze(2)[:,:,-1],
            pred_mask.squeeze(2)[:,:,-1]
        )
        pred_last_acc = utils.pred_test_last(
            device,
            pred_out.squeeze(3).transpose(-1,-2),
            pred_true,
            pred_pred_mask.squeeze(2).bool(),
            last_step.unsqueeze(1).unsqueeze(1),
            pred_pred_mask.squeeze(2)[:,:,-1]
        )
        return [pred_last_acc]

    def pred_mean_acc_fn(self, device, pred_out, pred_true, pred_mask, term_mask):
        pred_pred_mask = (pred_true != float('inf')).long()
        pred_mask = pred_mask.long()
        pred_true = self.process_pred_true(
            device,
            pred_true.squeeze(2),
            pred_mask.squeeze(2)
        )
        pred_mean_acc = utils.pred_test_mean(
            device,
            pred_out.squeeze(3).transpose(-1,-2),
            pred_true.squeeze(2),
            pred_pred_mask.squeeze(2).bool(),
            term_mask,
            pred_mask.squeeze(2)
        )
        return [pred_mean_acc]
