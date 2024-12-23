"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import TP_RGCNLayer as Layer
from .layers import RGCNBasisLayer as RGCNLayer

from .aggregators import Transpooling, Graphpooling, SumAggregator, Simple_Graphpooling


class TP_RGCN(nn.Module):
    def __init__(self, params):
        super(TP_RGCN, self).__init__()

        # data parameters
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        
        # model parameters
        self.inp_dim = params.inp_dim
        self.emb_dim = params.emb_dim
        self.att_dim = params.att_dim
        self.num_heads = params.num_heads
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        self.dropout = params.dropout
        self.edge_dropout = params.edge_dropout
        self.device = params.device

        # create tp-gcn layers
        self.build_model()


    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        transpooling_i2h = Transpooling(self.num_rels, self.inp_dim, self.emb_dim, 
                                        self.att_dim, self.num_heads)
        RGCNi2h = Layer(self.inp_dim, self.emb_dim, self.aug_num_rels, transpooling_i2h, is_input_layer=True, 
                    num_bases=self.num_bases, activation=F.relu, dropout=self.dropout, edge_dropout=self.edge_dropout)
        self.layers.append(RGCNi2h)
        
        # h2h
        for _ in range(self.num_hidden_layers - 1):
            transpooling_h2h = Transpooling(self.num_rels, self.emb_dim, self.emb_dim, 
                                        self.att_dim, self.num_heads)
            h2h = Layer(self.emb_dim, self.emb_dim, self.aug_num_rels, transpooling_h2h, is_input_layer=False, 
                    num_bases=self.num_bases, activation=F.relu, dropout=self.dropout, edge_dropout=self.edge_dropout)
            self.layers.append(h2h)


    def forward(self, g):
        for layer in self.layers:
            layer(g)
        return g.ndata.pop('h')


class RGCN_graphpool(nn.Module):
    def __init__(self, params):
        super(RGCN_graphpool, self).__init__()

        # data parameters
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        
        # model parameters
        self.inp_dim = params.inp_dim
        self.emb_dim = params.emb_dim
        self.att_dim = params.att_dim
        self.attn_rel_emb_dim = params.attn_rel_emb_dim
        
        self.num_heads = params.num_heads
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        
        self.dropout = params.G_dropout
        self.edge_dropout = params.edge_dropout
        self.has_attn = params.has_attn
        self.graphpooling_mode = params.graphpooling_mode
        
        self.device = params.device
        
        if self.has_attn:
            self.attn_rel_emb = nn.Embedding(self.num_rels, self.attn_rel_emb_dim, sparse=False)
        else:
            self.attn_rel_emb = None
            
        self.aggregator = SumAggregator()

        # create tp-gcn layers
        self.build_model()


    def build_model(self):
        self.gnn_layers = nn.ModuleList()
        # i2h
        i2h = RGCNLayer(self.inp_dim,
                        self.emb_dim,
                        self.aggregator,
                        self.attn_rel_emb_dim,
                        self.aug_num_rels,
                        self.num_bases,
                        activation=F.relu,
                        dropout=self.dropout,
                        edge_dropout=self.edge_dropout,
                        is_input_layer=True,
                        has_attn=self.has_attn)
        self.gnn_layers.append(i2h)
        # h2h
        for _ in range(self.num_hidden_layers - 1):
            h2h = RGCNLayer(self.emb_dim,
                            self.emb_dim,
                            self.aggregator,
                            self.attn_rel_emb_dim,
                            self.aug_num_rels,
                            self.num_bases,
                            activation=F.relu,
                            dropout=self.dropout,
                            edge_dropout=self.edge_dropout,
                            has_attn=self.has_attn)
            self.gnn_layers.append(h2h)
        # graph pooling
        if self.graphpooling_mode == 'GP':
            self.graphpooling = Graphpooling(self.emb_dim, self.att_dim, 
                                             self.num_heads, self.num_hidden_layers)
        elif self.graphpooling_mode == 'sGP':
            self.graphpooling = Simple_Graphpooling(self.emb_dim, self.att_dim, 
                                                    self.num_heads, self.num_hidden_layers)


    def forward(self, bg):
        for layer in self.gnn_layers:
            layer(bg, self.attn_rel_emb)
            
        g_out = torch.cat([self.graphpooling(g).unsqueeze(0) for g in dgl.unbatch(bg)], dim=0)
        return g_out