import torch
import torch.nn as nn
import torch.nn.functional as fn

class MPNNlayer(nn.Module):
    """
    Message Passing Neural Network variant (Gilmer et al., ICML 2017)
    """
    def __init__(self, device, enc_dim, edge_dim, latent_dim, bias=True, gru=False, actv=None):
        super(MPNNlayer, self).__init__()

        self.dev = device
        self.zdim = enc_dim
        self.hdim = latent_dim
        self.edim = edge_dim

        self.bias = bias

        if actv is None:
            self.message = nn.Sequential(
                nn.Linear(2*enc_dim+edge_dim, latent_dim, bias=bias),
            )
        else:
            self.message = nn.Sequential(
                nn.Linear(2*enc_dim+edge_dim, latent_dim, bias=bias),
                actv
            )

        # todo upgrade to pytorch 1.7 to use amax!!
        self.aggregate = torch.max
        if gru:
            self.gru = nn.GRUCell(latent_dim, latent_dim, bias=bias)
        else:
            self.gru = None
        if actv is None:
            self.update = nn.Sequential(
                nn.Linear(enc_dim+latent_dim, latent_dim, bias=bias),
            )
        else:
            self.update = nn.Sequential(
                nn.Linear(enc_dim+latent_dim, latent_dim, bias=bias),
                actv
            )


        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        # TF Keras initializer
        nn.init.xavier_uniform_(self.message[0].weight.data)
        if self.bias:
            nn.init.zeros_(self.message[0].bias.data)
        nn.init.xavier_uniform_(self.update[0].weight.data)
        if self.bias:
            nn.init.zeros_(self.update[0].bias.data)
        # TF Lattice initializer
        # nn.init.uniform_(self.message[0].weight.data,-0.05,0.05)
        # nn.init.uniform_(self.message[0].bias.data,-0.05,0.05)
        # nn.init.uniform_(self.update[0].weight.data,-0.05,0.05)
        # nn.init.uniform_(self.update[0].bias.data,-0.05,0.05)

    def forward(self, z, e_feat, adj, h=None):
        """
        z: BATCH x NUM_NODES x NUM_NODE_FEATURES (B x N x M)
        e_feat: BATCH x NUM_NODES x NUM_NODES x NUM_EDGE_FEATURES (B x N x N x E)
        adj: BATCH x NUM_NODES x NUM_NODES (B x N x N)
        """

        z_i = z.unsqueeze(2).expand(-1,-1,adj.shape[1],-1)
        z_j = z.unsqueeze(1).expand(-1,adj.shape[1],-1,-1)
        # (z_ij: B x N x N x (2M+E))
        if self.edim == 0:
            z_ij = torch.cat([z_i, z_j], dim=-1)
        else:
            z_ij = torch.cat([z_i, z_j, e_feat], dim=-1)
        shape = z_ij.shape

        # temp. change to ( BNN x M ) to pass it through the message neural network
        z_ij = z_ij.view(-1, shape[-1])
        msgs = self.message(z_ij)

        # mask value and operator depends on aggregator for max its float('-inf') with addition -> todo for other aggregators
        adj_mask = adj.float().masked_fill((adj==0).bool(), float('-inf')).unsqueeze(-1)
        adj_mask = adj_mask.masked_fill((adj==1).bool().unsqueeze(-1), 0.0)
        # changing it back for aggregation (msgs: B x N x N x M)
        msgs = msgs.view(*(shape[:-1]),self.hdim)
        # (agg_msgs: B x N x M)
        agg_msgs = self.aggregate(msgs+adj_mask, dim=2)[0]

        # (out_z: B x N x (2M))
        out_h = torch.cat([z, agg_msgs], dim=-1)
        # compactifying the new representation to the original size
        out_h = out_h.view(-1, self.zdim+self.hdim)
        out_h = self.update(out_h)
        if self.gru is not None:
            out_h = self.gru(out_h, h.view(-1, self.hdim))
        # final dimensions (out_z: B x N x M)
        out_h = out_h.view(shape[0], shape[1], self.hdim)
        return out_h

class MultiMPNN(nn.Module):
    """
    Message Passing Neural Network variant (Gilmer et al., ICML 2017)
    """
    def __init__(
            self,
            device,
            enc_dim,
            edge_dim,
            latent_dim,
            bias=True,
            gru=False,
            actv=None
    ):
        super(MultiMPNN, self).__init__()

        self.dev = device
        self.zdim = enc_dim
        self.hdim = latent_dim
        self.edim = edge_dim

        self.bias = bias

        if actv is None:
            self.message = nn.Sequential(
                nn.Linear(2*enc_dim+edge_dim, latent_dim, bias=bias),
            )
        else:
            self.message = nn.Sequential(
                nn.Linear(2*enc_dim+edge_dim, latent_dim, bias=bias),
                actv
            )

        self.aggregate = torch.max
        if gru:
            self.gru = nn.GRUCell(latent_dim, latent_dim, bias=bias)
        else:
            self.gru = None
        if actv is None:
            self.update = nn.Sequential(
                nn.Linear(enc_dim+latent_dim, latent_dim, bias=bias),
            )
        else:
            self.update = nn.Sequential(
                nn.Linear(enc_dim+latent_dim, latent_dim, bias=bias),
                actv
            )


        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        # TF Keras initializer
        nn.init.xavier_uniform_(self.message[0].weight.data)
        if self.bias:
            nn.init.zeros_(self.message[0].bias.data)
        nn.init.xavier_uniform_(self.update[0].weight.data)
        if self.bias:
            nn.init.zeros_(self.update[0].bias.data)
        # # Identity initializer
        # nn.init.eye_(self.message[0].weight.data)
        # if self.bias:
        #     nn.init.zeros_(self.message[0].bias.data)
        # nn.init.eye_(self.update[0].weight.data)
        if self.bias:
            nn.init.zeros_(self.update[0].bias.data)
        # TF Lattice initializer
        # nn.init.uniform_(self.message[0].weight.data,-0.05,0.05)
        # nn.init.uniform_(self.message[0].bias.data,-0.05,0.05)
        # nn.init.uniform_(self.update[0].weight.data,-0.05,0.05)
        # nn.init.uniform_(self.update[0].bias.data,-0.05,0.05)

    def forward(self, zs, e_feat, adj, encoders, h=None):
        """
        z: BATCH x NUM_NODES x NUM_NODE_FEATURES (B x N x M)
        e_feat: BATCH x NUM_NODES x NUM_NODES x NUM_EDGE_FEATURES (B x N x N x E)
        adj: BATCH x NUM_NODES x NUM_NODES (B x N x N)
        """

        out = []
        for z, enc in zip(zs, encoders):
            z_i = z.unsqueeze(2).expand(-1,-1,adj.shape[1],-1)
            z_j = z.unsqueeze(1).expand(-1,adj.shape[1],-1,-1)
            # (z_ij: B x N x N x (2M+E))
            z_ij = torch.cat([z_i, z_j, e_feat], dim=-1)
            shape = z_ij.shape

            # temp. change to ( BNN x M ) to pass it through the message neural network
            z_ij = z_ij.view(-1, shape[-1])
            if enc is not None:
                z_ij = enc(z_ij)
            msgs = self.message(z_ij)

            # mask value and operator depends on aggregator for max its float('-inf') with addition -> todo for other aggregators
            adj_mask = adj.float().masked_fill((adj==0).bool(), float('-inf')).unsqueeze(-1)
            adj_mask = adj_mask.masked_fill((adj==1).bool().unsqueeze(-1), 0.0)
            # changing it back for aggregation (msgs: B x N x N x M)
            msgs = msgs.view(*(shape[:-1]),self.hdim)
            # (agg_msgs: B x N x M)
            agg_msgs = self.aggregate(msgs+adj_mask, dim=2)[0]

            # (out_z: B x N x (2M))
            out_h = torch.cat([z[:,:,:self.zdim], agg_msgs], dim=-1)
            # compactifying the new representation to the original size
            out_h = out_h.view(-1, self.zdim+self.hdim)
            out_h = self.update(out_h)
            if self.gru is not None:
                out_h = self.gru(out_h, h.view(-1, self.hdim))
            # final dimensions (out_z: B x N x M)
            out_h = out_h.view(shape[0], shape[1], self.hdim)
            out.append(out_h)
        return out

