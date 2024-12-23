import torch
import torch.nn as nn
import torch.nn.functional as F


class MEAN(nn.Module):
    def __init__(self, params):
        super(MEAN, self).__init__()

        self.inp_dim = params.emb_dim
        self.emb_dim = params.emb_dim
        self.num_layers = params.num_gnn_layers
        self.dropout = params.dropout
        self.edge_dropout = params.edge_dropout
        # self.aggregator_type = params.gnn_agg_type
        self.device = params.device
        self.params = params
        
        self.num_ent = params.num_ent
        self.num_rel = params.num_rel
        self.entity2id = params.entity2id
        self.relation2id = params.relation2id

        # initialize basis weights for input and hidden layers
        # self.input_basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.emb_dim))
        # self.basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.emb_dim, self.emb_dim))

        # create rgcn layers
        self.build_model()
        self.initialize_features()
    
    
    def initialize_features(self):
        self.ent_embs = nn.Embedding(self.num_ent, self.emb_dim)
        self.rel_embs = nn.Embedding(self.num_rel, self.emb_dim)
        

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = MEANLayer(self.inp_dim, self.emb_dim, self.params)
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_layers - 1):
            h2h = MEANLayer(self.emb_dim, self.emb_dim, self.params)
            self.layers.append(h2h)

    def forward(self, g):
        ori_ent_ID = g.ndata['id']
        g.ndata['h'] = self.ent_embs(ori_ent_ID)
        for layer in self.layers:
            layer(g)
        
        ent_embs = self.ent_embs.weight.clone()
        ent_embs[ori_ent_ID] = g.ndata.pop('h')
        return ent_embs

    
    def score(self, triplets, ent_embs):
        h, r, t = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        h_embs = ent_embs[h]
        r_embs = self.rel_embs(r)
        t_embs = ent_embs[t]
        scores = h_embs + r_embs - t_embs
        scores = torch.norm(scores, 1, -1)
        
        return scores


class MEANLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, params, dropout=0.0):
        super(MEANLayer, self).__init__()
        self.params = params
        
        self.weight = nn.Parameter(torch.Tensor(params.num_rel, inp_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        self.batchnorm = nn.BatchNorm1d(self.params.emb_dim)
        
    
    def forward(self, g):
        
        def msg_func(edges):
            w = self.weight.index_select(0, edges.data['type'])
            msg = torch.relu(torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze(1))
            norm_msg = self.batchnorm(msg)
            return {'msg': norm_msg}
        
        def reduce_func(nodes):
            return {'h': torch.mean(nodes.mailbox['msg'], dim=1)}
        
        g.update_all(msg_func, reduce_func)