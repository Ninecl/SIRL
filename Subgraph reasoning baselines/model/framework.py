import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import mean_nodes
from .rgcn import RGCN
from .tprgcn import RGCN_graphpool
from .RTransformer import RTransformer


class Grail(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.emb_dim, sparse=False)

        if self.params.add_ht_emb:
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim + self.params.emb_dim, 1)

    def forward(self, data):
        g, rel_labels = data
        g.ndata['h'] = self.gnn(g)

        g_out = mean_nodes(g, 'repr')

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        if self.params.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               self.rel_emb(rel_labels)], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim), self.rel_emb(rel_labels)], dim=1)

        output = self.fc_layer(g_rep)
        return output


class ISE2(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id

        self.model = RGCN_graphpool(params)
        self.rel_emb = nn.Embedding(params.num_rels, params.emb_dim)
        self.ent_proj_emb = nn.Embedding(params.num_rels, params.emb_dim)
        
        if self.params.num_RT_layers > 0:
            self.RTransformer = RTransformer(params)
            self.scores_weight = nn.Parameter(torch.tensor([[0.1], [0.9]]))

        if self.params.add_ht_emb:
            self.fc_layer = nn.Linear(3 * params.num_gcn_layers * params.emb_dim + params.emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(params.num_gcn_layers * params.emb_dim + params.emb_dim, 1)


    def forward(self, data):
        # input data
        g, rel_labels, rfs = data
        
        # GP
        g_out = self.model(g)
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        if self.params.add_ht_emb:
            G_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               self.rel_emb(rel_labels)], dim=1)
        else:
            G_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim), self.rel_emb(rel_labels)], dim=1)

        G_scores = self.fc_layer(G_rep)
        
        # RT
        if self.params.num_RT_layers > 0:
            R_scores = self.RTransformer(rfs, rel_labels)
            # A_scores
            # if self.params.score_fusion == 'tanh':
            #     scores = torch.cat((torch.tanh(R_scores), torch.tanh(G_scores)), dim=1)
            # elif self.params.score_fusion == 'sigmoid':
            #     scores = torch.cat((torch.sigmoid(R_scores), torch.sigmoid(G_scores)), dim=1)
            # elif self.params.score_fusion == 'norm':
            #     scores = torch.cat((F.normalize(R_scores, p=1), F.normalize(G_scores, p=1)), dim=1)
            # else:
            #     scores = torch.cat((R_scores, G_scores), dim=1)
            scores = torch.cat((R_scores, G_scores), dim=1)
            A_scores = torch.mm(scores, self.scores_weight)
            # print(R_scores, G_scores, A_scores)
            return R_scores, G_scores, A_scores
        else:
            return G_scores
