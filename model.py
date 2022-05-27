import torch
import torch.nn as nn

from layer import MPNNlayer, MultiMPNN

def create_model(name, device, args, metadata):
    if name == 'NeuralExec':
        return NeuralExecutor(device,
                              sum(metadata['nodedim']),
                              max(metadata['edgedim']),
                              args['latentdim'],
                              args['encdim'],
                              pred=sum(metadata['pred']),
                              noise=args['noisedim']
                              )
    if name == 'NeuralExecGRU':
        return NeuralExecutor(device,
                              sum(metadata['nodedim']),
                              max(metadata['edgedim']),
                              args['latentdim'],
                              args['encdim'],
                              pred=sum(metadata['pred']),
                              gru=True,
                              noise=args['noisedim']
                              )
    if name == 'NeuralExec2':
        return NeuralExecutor2(device,
                              sum(metadata['nodedim']),
                              max(metadata['edgedim']),
                              args['latentdim'],
                              args['encdim'],
                              pred=sum(metadata['pred']),
                              noise=args['noisedim']
                              )
    if name == 'NeuralExec2detach':
        return NeuralExecutor2(device,
                               sum(metadata['nodedim']),
                               max(metadata['edgedim']),
                               args['latentdim'],
                               args['encdim'],
                               pred=sum(metadata['pred']),
                               noise=args['noisedim'],
                               term_detach=True
                              )
    if name == 'NeuralExec2Transfer':
        return NeuralExecutor2Transfer(device,
                                       sum(metadata['nodedim']),
                                       max(metadata['edgedim']),
                                       args['latentdim'],
                                       args['encdim'],
                                       pred=sum(metadata['pred']),
                                       noise=args['noisedim'],
                                       term_detach=True
                                       )
    if name == 'NeuralExec2Freeze':
        return NeuralExecutor2Freeze(device,
                                     sum(metadata['nodedim']),
                                     max(metadata['edgedim']),
                                     args['latentdim'],
                                     args['encdim'],
                                     pred=sum(metadata['pred']),
                                     noise=args['noisedim'],
                                     term_detach=True
                                     )
    if name == 'NeuralExec3':
        return NeuralExecutor3(device,
                              metadata['nodedim'],
                              metadata['edgedim'],
                              args['latentdim'],
                              args['encdim'],
                              pred=metadata['pred'],
                              noise=args['noisedim']
                              )
    if name == 'NeuralExec3Freeze':
        return NeuralExecutor3Freeze(device,
                              metadata['nodedim'],
                              metadata['edgedim'],
                              args['latentdim'],
                              args['encdim'],
                              pred=metadata['pred'],
                              noise=args['noisedim'],
                              term_detach=True
                              )
    if name == 'NeuralExec3Transfer':
        return NeuralExecutor3Transfer(device,
                              metadata['nodedim'],
                              metadata['edgedim'],
                              args['latentdim'],
                              args['encdim'],
                              pred=metadata['pred'],
                              noise=args['noisedim'],
                              term_detach=True
                              )
    if name == 'NeuralExec4':
        return NeuralExecutor4(device,
                              metadata['nodedim'],
                              metadata['edgedim'],
                              args['latentdim'],
                              args['encdim'],
                              pred=metadata['pred'],
                              noise=args['noisedim']
                              )


class NeuralExecutor(nn.Module):
    """
    The model proposed in Neural Execution of Graph Algorithms (Velickovic et al., ICLR 2020)
    """
    def __init__(self, device, node_dim, edge_dim, latent_dim, enc_dim, pred=0, bias=True, gru=False, noise=0):
        super(NeuralExecutor, self).__init__()

        self.dev = device
        self.ndim = node_dim
        self.edim = edge_dim
        self.hdim = latent_dim
        self.encdim = enc_dim
        self.noise = noise

        self.noise_gen = torch.distributions.uniform.Uniform(0.0, 1.0)

        self.pred = pred

        self.temp = 1.0

        self.processor = MPNNlayer(device, enc_dim+noise, enc_dim, latent_dim, bias=bias, gru=gru, actv=nn.ReLU())
        self.node_encoder =  nn.Sequential(
                nn.Linear(latent_dim+node_dim, enc_dim, bias=bias)
                )
        self.edge_encoder =  nn.Sequential(
                nn.Linear(max(edge_dim, 1), enc_dim, bias=bias)
        )
        self.decoder =  nn.Sequential(
                nn.Linear(latent_dim+enc_dim, node_dim, bias=bias)
                )
        if pred != 0:
            self.predecessor = nn.ModuleList([ nn.Sequential(
                nn.Linear(2*latent_dim+enc_dim, 1, bias=bias)
            ).to(device) for _ in range(pred)])
        else:
            self.predecessor = []

        self.termination = nn.Linear(latent_dim, 1, bias=bias)
        self.gru = gru

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        self.processor.reset_parameters()
        # TF Keras init
        nn.init.xavier_uniform_(self.node_encoder[0].weight.data)
        nn.init.zeros_(self.node_encoder[0].bias.data)
        nn.init.xavier_uniform_(self.edge_encoder[0].weight.data)
        nn.init.zeros_(self.edge_encoder[0].bias.data)
        nn.init.xavier_uniform_(self.decoder[0].weight.data)
        nn.init.zeros_(self.decoder[0].bias.data)
        if self.pred != 0:
            for p in self.predecessor:
                nn.init.xavier_uniform_(p[0].weight.data)
                nn.init.zeros_(p[0].bias.data)
        nn.init.xavier_uniform_(self.termination.weight.data)
        nn.init.zeros_(self.termination.bias.data)
        # TF Lattice init
        # nn.init.uniform_(self.encoder.weight.data,-0.05,0.05)
        # nn.init.uniform_(self.encoder.bias.data,-0.05,0.05)
        # nn.init.uniform_(self.decoder.weight.data,-0.05,0.05)
        # nn.init.uniform_(self.decoder.bias.data,-0.05,0.05)
        # nn.init.uniform_(self.predecessor[0].weight.data,-0.05,0.05)
        # nn.init.uniform_(self.predecessor[0].bias.data,-0.05,0.05)
        # nn.init.uniform_(self.termination.weight.data,-0.05,0.05)
        # nn.init.uniform_(self.termination.bias.data,-0.05,0.05)
        self.temp = 1.0

    def forward(self, x, h, adj, e_feat):
        # add selfloops to graph
        adj = adj + torch.eye(adj.shape[1], device=self.dev).unsqueeze(0).expand_as(adj).long()

        if self.edim == 0:
            e_feat = adj.float()
        # number of nodes in the graph
        nnodes = adj.shape[1]

        z = self.node_encoder(torch.cat([x,h], dim=-1))
        noise = self.noise_gen.sample((x.shape[0], nnodes, self.noise)).to(self.dev)
        z_noise = torch.cat([z,noise], dim=-1)
        enc_e_feat = self.edge_encoder(e_feat.view(-1,max(self.edim, 1))).view(-1,nnodes,nnodes, self.encdim)
        if self.gru:
            new_h = self.processor(z_noise, enc_e_feat, adj, h)
        else:
            new_h = self.processor(z_noise, enc_e_feat, adj)
        new_x = self.decoder(torch.cat([z,new_h], dim=-1))
        tau = self.termination(torch.mean(new_h, dim=1))

        if self.pred != 0:
            p = []
            for pred_net in self.predecessor:
                # predicting the predecessor by computing attention over the edges of the graph
                new_h_i = new_h.unsqueeze(2).expand(-1,-1,adj.shape[1],-1)
                new_h_j = new_h.unsqueeze(1).expand(-1,adj.shape[1],-1,-1)
                new_h_ij = torch.cat([new_h_i, new_h_j, enc_e_feat], dim=-1)
                shape = new_h_ij.shape

                # temp. change to ( BNN x M ) to pass it through the predecessor neural network
                new_h_ij = new_h_ij.view(-1, shape[-1])
                p_i = pred_net(new_h_ij).view(*shape[:-1])

                # ensuring we only predict over neighbours
                mask = adj
                p_i = p_i.masked_fill(~mask.bool(), float('-inf'))
                p.append(p_i)
            p = torch.stack(p, dim=-1)
        else:
            p = torch.zeros((x.shape[0], x.shape[1], x.shape[1],1), device=self.dev)
        return new_x, p.squeeze(), tau, new_h

    def run_alg(self, x, adj, e_feat, max_steps, step_transition):
        b_size = x.shape[0]
        n_nodes = x.shape[1]
        model_steps = torch.zeros((b_size, n_nodes, max_steps+1, self.ndim), device=self.dev)
        tau = torch.zeros((b_size, max_steps), device=self.dev).unsqueeze(1)
        p = torch.zeros((b_size, n_nodes, max_steps, n_nodes, max(self.pred, 1)), device=self.dev).squeeze(dim=-1)
        h = torch.zeros((b_size, n_nodes, self.hdim), device=self.dev)

        model_steps[:,:,0,:] = x
        n_steps = 0
        for i in range(max_steps):
            model_steps[:, :,i+1, :], p[:,:,i,:], tau[:,:,i], h = step_transition(model_steps[:,:,i,:],
                                                                                      *self(model_steps[:,:,i,:], h, adj, e_feat),
                                                                                  self.temp
                                                                                  )
            n_steps = n_steps+1
            # if (torch.sigmoid(tau[:,:,i].squeeze()) > 0.5).all():
            #     break

        return n_steps, model_steps, tau, p.squeeze(dim=-1)


class NeuralExecutor2(nn.Module):
    """
    The model proposed in Neural Execution of Graph Algorithms (Velickovic et al., ICLR 2020)
    """
    def __init__(self, device, node_dim, edge_dim, latent_dim, enc_dim, pred=0, bias=False, gru=False, noise=0, term_detach=False):
        super(NeuralExecutor2, self).__init__()

        self.dev = device
        self.ndim = node_dim
        self.edim = edge_dim
        self.hdim = latent_dim
        self.encdim = enc_dim
        self.noise = noise

        self.noise_gen = torch.distributions.uniform.Uniform(0.0, 1.0)

        self.pred = pred
        self.bias = bias

        self.temp = 1.0
        self.term_detach = term_detach

        self.processor = MPNNlayer(device, enc_dim+noise, enc_dim, latent_dim, bias=bias, gru=gru, actv=None)
        self.node_encoder =  nn.Sequential(
                nn.Linear(latent_dim+node_dim, enc_dim, bias=bias)
                )
        self.edge_encoder =  nn.Sequential(
                nn.Linear(max(edge_dim, 1), enc_dim, bias=bias)
        )
        self.decoder =  nn.Sequential(
                nn.Linear(latent_dim+enc_dim, node_dim, bias=bias)
                )
        if pred != 0:
            self.predecessor = nn.ModuleList([ nn.Sequential(
                nn.Linear(2*latent_dim+enc_dim, 1, bias=bias)
            ).to(device) for _ in range(pred)])

        self.termination_mpnn = MPNNlayer(device, latent_dim, latent_dim, latent_dim, bias=bias, gru=gru, actv=None)
        self.termination = nn.Linear(latent_dim, 1, bias=bias)
        self.gru = gru

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        self.processor.reset_parameters()
        # TF Keras init
        nn.init.xavier_uniform_(self.node_encoder[0].weight.data)
        if self.bias:
            nn.init.zeros_(self.node_encoder[0].bias.data)
        nn.init.xavier_uniform_(self.edge_encoder[0].weight.data)
        if self.bias:
            nn.init.zeros_(self.edge_encoder[0].bias.data)
        nn.init.xavier_uniform_(self.decoder[0].weight.data)
        if self.bias:
            nn.init.zeros_(self.decoder[0].bias.data)
        if self.pred != 0:
            for p in self.predecessor:
                nn.init.xavier_uniform_(p[0].weight.data)
                if self.bias:
                    nn.init.zeros_(p[0].bias.data)
        nn.init.xavier_uniform_(self.termination.weight.data)
        if self.bias:
            nn.init.zeros_(self.termination.bias.data)
        # TF Lattice init
        # nn.init.uniform_(self.encoder.weight.data,-0.05,0.05)
        # nn.init.uniform_(self.encoder.bias.data,-0.05,0.05)
        # nn.init.uniform_(self.decoder.weight.data,-0.05,0.05)
        # nn.init.uniform_(self.decoder.bias.data,-0.05,0.05)
        # nn.init.uniform_(self.predecessor[0].weight.data,-0.05,0.05)
        # nn.init.uniform_(self.predecessor[0].bias.data,-0.05,0.05)
        # nn.init.uniform_(self.termination.weight.data,-0.05,0.05)
        # nn.init.uniform_(self.termination.bias.data,-0.05,0.05)
        self.temp = 1.0

    def forward(self, x, h, adj, e_feat):
        # add selfloops to graph
        adj = adj + torch.eye(adj.shape[1], device=self.dev).unsqueeze(0).expand_as(adj).long()

        if self.edim == 0:
            e_feat = adj.float()
        # number of nodes in the graph
        nnodes = adj.shape[1]

        z = self.node_encoder(torch.cat([x,h], dim=-1))
        noise = self.noise_gen.sample((x.shape[0], nnodes, self.noise)).to(self.dev)
        z_noise = torch.cat([z,noise], dim=-1)
        enc_e_feat = self.edge_encoder(e_feat.view(-1,max(self.edim, 1))).view(-1,nnodes,nnodes, self.encdim)
        if self.gru:
            new_h = self.processor(z_noise, enc_e_feat, adj, h)+z
        else:
            new_h = self.processor(z_noise, enc_e_feat, adj)+z
        new_x = self.decoder(torch.cat([z,new_h], dim=-1))
        if self.term_detach:
            tau_node = self.termination_mpnn(new_h.detach(), enc_e_feat, adj)
        else:
            tau_node = self.termination_mpnn(new_h, enc_e_feat, adj)
        tau = self.termination(torch.mean(tau_node, dim=1))

        if self.pred != 0:
            p = []
            for pred_net in self.predecessor:
                # predicting the predecessor by computing attention over the edges of the graph
                new_h_i = new_h.unsqueeze(2).expand(-1,-1,adj.shape[1],-1)
                new_h_j = new_h.unsqueeze(1).expand(-1,adj.shape[1],-1,-1)
                new_h_ij = torch.cat([new_h_i, new_h_j, enc_e_feat], dim=-1)
                shape = new_h_ij.shape

                # temp. change to ( BNN x M ) to pass it through the predecessor neural network
                new_h_ij = new_h_ij.view(-1, shape[-1])
                p_i = pred_net(new_h_ij).view(*shape[:-1])

                # ensuring we only predict over neighbours
                mask = adj
                p_i = p_i.masked_fill(~mask.bool(), float('-inf'))
                p.append(p_i)
            p = torch.stack(p, dim=-1)
        else:
            p = torch.zeros((x.shape[0], x.shape[1], x.shape[1],1), device=self.dev)
        return new_x, p.squeeze(), tau, new_h

    def run_alg(self, x, adj, e_feat, max_steps, step_transition):
        b_size = x.shape[0]
        n_nodes = x.shape[1]
        model_steps = torch.zeros((b_size, n_nodes, max_steps+1, self.ndim), device=self.dev)
        tau = torch.zeros((b_size, max_steps), device=self.dev).unsqueeze(1)
        p = torch.zeros((b_size, n_nodes, max_steps, n_nodes, max(self.pred, 1)), device=self.dev).squeeze(dim=-1)
        h = torch.zeros((b_size, n_nodes, self.hdim), device=self.dev)

        model_steps[:,:,0,:] = x
        n_steps = 0
        for i in range(max_steps):
            model_steps[:, :,i+1, :], p[:,:,i,:], tau[:,:,i], h = step_transition(model_steps[:,:,i,:],
                                                                                      *self(model_steps[:,:,i,:], h, adj, e_feat),
                                                                                  self.temp
                                                                                  )
            n_steps = n_steps+1
            # if (torch.sigmoid(tau[:,:,i].squeeze()) > 0.5).all():
            #     break

        return n_steps, model_steps, tau, p.squeeze(dim=-1)


class ParallelProcessors(nn.Module):
    def __init__(self, n_procs, device, enc_dim, edge_dim, latent_dim, bias=True, gru=False, actv=None, noise=0):
        super(ParallelProcessors, self).__init__()
        self.dev = device
        self.n_procs = n_procs
        self.processors = nn.ModuleList([
            MPNNlayer(device, enc_dim+noise, enc_dim, latent_dim, bias=bias, gru=gru, actv=actv) for _ in range(n_procs)
        ])

        self.processor_attention = nn.ParameterList([
            nn.Parameter(torch.ones((1), device=device)) for _ in range(n_procs)
        ])

    def reset_parameters(self):
        for mpnn in self.processors:
            mpnn.reset_parameters()
        self.processor_attention = nn.ParameterList([
            nn.Parameter(0.5*torch.ones((1), device=self.dev)) for _ in range(self.n_procs)
        ])

    def forward(self, z, e_feat, adj, h=None):
        out = 0
        for mpnn, coef in zip(self.processors, self.processor_attention):
            out += coef * mpnn(z, e_feat, adj, h)
        return out


class NeuralExecutor2Transfer(NeuralExecutor2):
    def __init__(self, device, node_dim, edge_dim, latent_dim, enc_dim, pred=0, bias=False, gru=False, noise=0, term_detach=False):
        super(NeuralExecutor2Transfer, self).__init__(device, node_dim, edge_dim, latent_dim, enc_dim, pred, bias, gru, noise, term_detach)
        self.processor = ParallelProcessors(2, device, enc_dim+noise, enc_dim, latent_dim, bias=bias, gru=gru, actv=None)


class NeuralExecutor2Freeze(NeuralExecutor2):
    def __init__(self, device, node_dim, edge_dim, latent_dim, enc_dim, pred=0, bias=False, gru=False, noise=0, term_detach=False):
        super(NeuralExecutor2Freeze, self).__init__(device, node_dim, edge_dim, latent_dim, enc_dim, pred, bias, gru, noise, term_detach)
        self.processor.requires_grad_(False)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias,0.01)

class Encoder(nn.Module):

    def __init__(self,
                 device,
                 node_dim,
                 latent_dim,
                 enc_dim):
        super(Encoder, self).__init__()
        self.non_linear = nn.Sequential(
            nn.Linear(2*latent_dim+2*node_dim+enc_dim, 2*latent_dim+enc_dim, bias=False),
            nn.ReLU(),
            nn.Linear(2*latent_dim+enc_dim, 2*latent_dim+enc_dim, bias=False),
        ).to(device)
        self.linear = nn.Sequential(
            nn.Linear(2*latent_dim+2*node_dim+enc_dim, 2*latent_dim+enc_dim, bias=False),
        ).to(device)

    def reset_parameters(self):
        self.linear.apply(init_weights)
        self.non_linear.apply(init_weights)

    def forward(self, x):
        return self.linear(x)+self.non_linear(x)

class NeuralExecutor3(nn.Module):
    """
    Based on the model proposed in Neural Execution of Graph Algorithms (Velickovic et al., ICLR 2020)
    """
    def __init__(self, device, node_dim, edge_dim, latent_dim, enc_dim, pred=0, bias=False, gru=False, noise=0, term_detach=False):
        super(NeuralExecutor3, self).__init__()

        self.dev = device
        self.ndim = node_dim
        self.edim = edge_dim
        self.hdim = latent_dim
        self.encdim = enc_dim
        self.noise = noise

        self.noise_gen = torch.distributions.uniform.Uniform(0.0, 1.0)

        self.pred = pred
        self.bias = bias

        self.temp = 1.0
        self.term_detach = term_detach

        self.processor = MultiMPNN(device, enc_dim, enc_dim, latent_dim, bias=bias, gru=gru, actv=None)
        self.edge_encoder = nn.ModuleList([ nn.Sequential(
                nn.Linear(max(edim, 1), enc_dim, bias=bias)
        ).to(device) for edim in edge_dim ])
        encoder = []
        for ndim in node_dim:
            encoder.append(
                Encoder(device,ndim,latent_dim,enc_dim).to(device)
            )
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList([ nn.Sequential(
                nn.Linear(latent_dim+enc_dim+ndim, ndim, bias=bias)
                ).to(device) for ndim in node_dim ])
        predecessor = []
        for pdim in pred:
            if pdim == 0:
                predecessor.append(None)
            else:
                predecessor.append(
                    nn.Sequential(
                        nn.Linear(2*latent_dim+enc_dim, pdim, bias=bias)
                    ).to(device)
                )
        self.predecessor = nn.ModuleList(predecessor)
        self.termination_mpnn = MPNNlayer(device, latent_dim, latent_dim, latent_dim, bias=bias, gru=gru, actv=None)
        self.termination = nn.Linear(latent_dim, 1, bias=bias)
        self.gru = gru

        self.reset_parameters()
        self.to(device)


    def reset_parameters(self):
        self.processor.reset_parameters()
        self.termination_mpnn.reset_parameters()
        # TF Keras init
        for i in range(len(self.encoder)):
            self.encoder[i].reset_parameters()
            nn.init.xavier_uniform_(self.edge_encoder[i][0].weight.data)
            if self.bias:
                nn.init.zeros_(self.edge_encoder[i][0].bias.data)
            nn.init.xavier_uniform_(self.decoder[i][0].weight.data)
            if self.bias:
                nn.init.zeros_(self.decoder[i][0].bias.data)
        if self.pred != 0:
            for p in self.predecessor:
                if p is not None:
                    nn.init.xavier_uniform_(p[0].weight.data)
                    if self.bias:
                        nn.init.zeros_(p[0].bias.data)
        nn.init.xavier_uniform_(self.termination.weight.data)
        if self.bias:
            nn.init.zeros_(self.termination.bias.data)

        # identity init
        # for i in range(len(self.node_encoder)):
        #     nn.init.eye_(self.node_encoder[i][0].weight.data)
        #     if self.bias:
        #         nn.init.zeros_(self.node_encoder[i][0].bias.data)
        #     nn.init.eye_(self.edge_encoder[i][0].weight.data)
        #     if self.bias:
        #         nn.init.zeros_(self.edge_encoder[i][0].bias.data)
        #     nn.init.eye_(self.decoder[i][0].weight.data)
        #     if self.bias:
        #         nn.init.zeros_(self.decoder[i][0].bias.data)
        # if self.pred != 0:
        #     for p in self.predecessor:
        #         if p is not None:
        #             nn.init.eye_(p[0].weight.data)
        #             if self.bias:
        #                 nn.init.zeros_(p[0].bias.data)
        # nn.init.eye_(self.termination.weight.data)
        # if self.bias:
        #     nn.init.zeros_(self.termination.bias.data)
        self.temp = 1.0


    def forward(self, xs, hs, adj, e_feat):
        # add selfloops to graph
        adj = adj + torch.eye(adj.shape[1], device=self.dev).unsqueeze(0).expand_as(adj).long()

        if self.edim == 0:
            e_feat = adj.float()
        # number of nodes in the graph
        nnodes = adj.shape[1]

        x_offset = 0
        h_offset = 0
        new_x = []
        all_h = []
        p = []
        for ndim, edim, edge_enc, dec, pred, enc in zip(self.ndim, self.edim, self.edge_encoder, self.decoder, self.predecessor, self.encoder):
            x = xs[:,:,x_offset:x_offset+ndim]
            h = hs[:,:,h_offset:h_offset+self.hdim]

            x_offset += ndim
            h_offset += self.hdim

            z = torch.cat([x,h], dim=-1)
            enc_e_feat = edge_enc(e_feat.view(-1,max(edim, 1))).view(-1,nnodes,nnodes, self.encdim)
            if self.gru:
                new_h = self.processor([z], enc_e_feat, adj, [enc], h=[h])[0]
            else:
                new_h = self.processor([z], enc_e_feat, adj, [enc])[0]
            new_x += [dec(torch.cat([z,new_h], dim=-1))]
            all_h += [new_h]

            if pred is not None:
                # predicting the predecessor by computing attention over the edges of the graph
                new_h_i = new_h.unsqueeze(2).expand(-1,-1,adj.shape[1],-1)
                new_h_j = new_h.unsqueeze(1).expand(-1,adj.shape[1],-1,-1)
                new_h_ij = torch.cat([new_h_i, new_h_j, enc_e_feat], dim=-1)
                shape = new_h_ij.shape

                # temp. change to ( BNN x M ) to pass it through the predecessor neural network
                new_h_ij = new_h_ij.view(-1, shape[-1])
                p_i = pred(new_h_ij).view(*shape[:-1])

                # ensuring we only predict over neighbours
                mask = adj
                p_i = p_i.masked_fill(~mask.bool(), float('-inf'))
                p += [p_i]
        new_x = torch.cat(new_x, dim=-1)
        if len(p) == 0:
            p = None
        elif len(p) == 1:
            p = p[0].unsqueeze(-1)
        else:
            p = torch.stack(p, dim=-1)
        cat_h = torch.cat(all_h, dim=-1)
        stack_h = torch.stack(all_h,dim=-1)
        if self.term_detach:
            tau_in = torch.mean(stack_h.detach(), dim=-1)
        else:
            tau_in = torch.mean(stack_h, dim=-1)
        tau_node = self.termination_mpnn(tau_in, enc_e_feat, adj)
        tau = self.termination(torch.mean(tau_node, dim=1))
        return new_x, p, tau, cat_h

    def run_alg(self, x, adj, e_feat, max_steps, step_transition):
        b_size = x.shape[0]
        n_nodes = x.shape[1]
        model_steps = torch.zeros((b_size, n_nodes, max_steps+1, self.ndim), device=self.dev)
        tau = torch.zeros((b_size, max_steps), device=self.dev).unsqueeze(1)
        p = torch.zeros((b_size, n_nodes, max_steps, n_nodes, max(self.pred, 1)), device=self.dev).squeeze(dim=-1)
        h = torch.zeros((b_size, n_nodes, self.hdim), device=self.dev)

        model_steps[:,:,0,:] = x
        n_steps = 0
        for i in range(max_steps):
            model_steps[:, :,i+1, :], p[:,:,i,:], tau[:,:,i], h = step_transition(model_steps[:,:,i,:],
                                                                                      *self(model_steps[:,:,i,:], h, adj, e_feat),
                                                                                  self.temp
                                                                                  )
            n_steps = n_steps+1
            # if (torch.sigmoid(tau[:,:,i].squeeze()) > 0.5).all():
            #     break

        return n_steps, model_steps, tau, p.squeeze(dim=-1)


class MultiParallelProcessors(nn.Module):
    def __init__(self, n_procs, device, enc_dim, edge_dim, latent_dim, bias=True, gru=False, actv=None, noise=0):
        super(MultiParallelProcessors, self).__init__()
        self.dev = device
        self.n_procs = n_procs
        self.processors = nn.ModuleList([
            MultiMPNN(device, enc_dim+noise, enc_dim, latent_dim, bias=bias, gru=gru, actv=actv) for _ in range(n_procs)
        ])

        self.processor_attention = nn.ParameterList([
            nn.Parameter(torch.ones((1), device=device)) for _ in range(n_procs)
        ])

    def reset_parameters(self):
        for mpnn in self.processors:
            mpnn.reset_parameters()
        self.processor_attention = nn.ParameterList([
            nn.Parameter(0.5*torch.ones((1), device=self.dev)) for _ in range(self.n_procs)
        ])

    def forward(self, z, e_feat, adj, enc, h=None):
        out = [0 for _ in range(len(z))]
        for mpnn, coef in zip(self.processors, self.processor_attention):
            out = [ out[i] + coef * val for i, val in enumerate(mpnn(z, e_feat, adj, enc, h)) ]
        return out


class NeuralExecutor3Transfer(NeuralExecutor3):
    def __init__(self, device, node_dim, edge_dim, latent_dim, enc_dim, pred=0, bias=False, gru=False, noise=0, term_detach=False):
        super(NeuralExecutor3Transfer, self).__init__(device, node_dim, edge_dim, latent_dim, enc_dim, pred, bias, gru, noise, term_detach)
        self.processor = MultiParallelProcessors(2, device, enc_dim+noise, enc_dim, latent_dim, bias=bias, gru=gru, actv=None)


class NeuralExecutor3Freeze(NeuralExecutor3):
    def __init__(self, device, node_dim, edge_dim, latent_dim, enc_dim, pred=0, bias=False, gru=False, noise=0, term_detach=False):
        super(NeuralExecutor3Freeze, self).__init__(device, node_dim, edge_dim, latent_dim, enc_dim, pred, bias, gru, noise, term_detach)
        self.processor.requires_grad_(False)


### Optimal model
def optimal_model(device, logger, metadata, algo):
    if algo == 'prims':
        model = NeuralExecutor(device,
                                metadata['nodedim'],
                                metadata['edgedim'],
                                2*metadata['nodedim'],
                                2*metadata['nodedim'],
                                pred=metadata['pred'],
                                noise=0
                                )
        ## termination can't be done with the current arch I think :(
        model.node_encoder[0].weight.data = torch.tensor([[1., 0., 0.], [1., 0., 0.]], device=model.dev)
        model.node_encoder[0].bias.data = torch.zeros((2*metadata['nodedim'],), device=model.dev)

        model.decoder[0].weight.data = torch.tensor([[0., 0., 1., 0.]], device=model.dev)
        model.decoder[0].bias.data = torch.zeros((metadata['nodedim']), device=model.dev)

        model.predecessor[0][0].weight.data = torch.tensor([[0., -1., 0., 1., 0., -1.]], device=model.dev)
        model.predecessor[0][0].bias.data = torch.zeros((metadata['nodedim']), device=model.dev)

        model.edge_encoder[0].weight.data = torch.tensor([[1.],[1.]], device=model.dev)
        model.edge_encoder[0].bias.data = torch.zeros((2*metadata['edgedim'],), device=model.dev)

        # processor
        model.processor.message[0].weight.data = torch.tensor([[0., 0., 1., 0., -1., 0.], [0., 0., 0., 0., 0., 0.]], device=model.dev)
        model.processor.message[0].bias.data = torch.zeros((2*metadata['nodedim'],), device=model.dev)

        model.processor.update[0].weight.data = torch.tensor([[-1., 0., 1., 0.], [0., 1., 0., 0.]], device=model.dev)
        model.processor.update[0].bias.data = torch.zeros((2*metadata['nodedim'],), device=model.dev)
    elif algo == 'dijkstra':
        ## not done, under construction
        model = NeuralExecutor(device,
                                metadata['nodedim'],
                                metadata['edgedim'],
                                2*metadata['nodedim'],
                                2*metadata['nodedim'],
                                pred=metadata['pred'],
                                noise=0
                                )
        ## termination can't be done with the current arch I think :(
        model.node_encoder[0].weight.data = torch.tensor([[1., 0., 0., 0., 0., 0.],
                                                          [0., 1., 0., 0., 0., 0.],
                                                          [1., 0., 0., 0., 0., 0.],
                                                          [0., 1., 0., 0., 0., 0.]], device=model.dev)
        model.node_encoder[0].bias.data = torch.zeros((2*metadata['nodedim'],), device=model.dev)

        model.decoder[0].weight.data = torch.tensor([[0., 0., 0., 0., 1., 0., 0., 0.],
                                                     [0., 0., 0., 0., 0., 1., 0., 0.]], device=model.dev)
        model.decoder[0].bias.data = torch.zeros((metadata['nodedim']), device=model.dev)

        model.predecessor[0][0].weight.data = torch.tensor([[0., 0.,-50., 0., 0., 0., 100., -1.,-1., 0., 0.,0.]], device=model.dev)
        model.predecessor[0][0].bias.data = torch.zeros((1), device=model.dev)

        model.edge_encoder[0].weight.data = torch.tensor([[1.],[1.], [1.], [1.]], device=model.dev)
        model.edge_encoder[0].bias.data = torch.zeros((4*metadata['edgedim'],), device=model.dev)

        # processor
        model.processor.message[0].weight.data = torch.tensor([[0., 0., 0., 0.,100., -1., 0., 0.,-1., 0., 0.,0.],
                                                               [0., 0., 0., 0.,100., -1., 0., 0.,-1., 0., 0.,0.],
                                                               [0., 0., 0., 0., 0.,  0., 0., 0., 0., 0., 0.,0.],
                                                               [0., 0., 0., 0., 0.,  0., 0., 0., 0., 0., 0.,0.],], device=model.dev)
        model.processor.message[0].bias.data = 0*torch.ones((2*metadata['nodedim'],), device=model.dev)

        model.processor.update[0].weight.data = torch.tensor([[-1., 0., 0., 0., 1., 0., 0., 0.],
                                                              [ 0., 0., 0., 0., 0.,-1., 0., 0.],
                                                              [ 1., 0., 0., 0., 0., 0., 0., 0.],
                                                              [ 0., 1., 0., 0., 0., 0., 0., 0.],], device=model.dev)
        model.processor.update[0].bias.data = torch.tensor([100., 100., 0., 0.], device=model.dev)
    else:
        logger.error("No optimal model for {} task".format(algo))
        sys.exit(1)
    return model
