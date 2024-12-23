import math
import torch

import torch.nn as nn
import torch.nn.functional as F


class RTransformer(nn.Module):
    
    def __init__(self, params):
        super(RTransformer, self).__init__()
        # parameters
        self.inp_dim = params.num_rels
        self.emb_dim = params.emb_dim * 3
        self.att_dim = params.att_dim
        self.num_rels = params.num_rels
        self.num_heads = params.num_heads
        self.num_layers = params.num_RT_layers
        self.rt_mode = params.relationalTransformer_mode
        
        # weights
        self.fcn = nn.Linear(self.emb_dim * 3, 1)
        self.rel_emb = nn.Embedding(params.num_rels, self.emb_dim)
        self.r_ent_emb = nn.Embedding(params.num_rels, self.emb_dim)
        
        # dropout
        self.drop_prob = params.R_dropout
        self.entity_drop_prob = params.entity_dropout
        if self.entity_drop_prob > 0:
            self.e_dropout = nn.Dropout(params.entity_dropout)
        else:
            self.e_dropout = None
        
        # ini
        self.build_model()
    

    def build_model(self):
        self.RTransformer_layers = nn.ModuleList()
        # i2h
        i2h = EncoderLayer(self.emb_dim, self.att_dim, self.num_heads, self.drop_prob, self.rt_mode)
        self.RTransformer_layers.append(i2h)
        # h2h
        for _ in range(0, self.num_layers-1):
            h2h = EncoderLayer(self.emb_dim, self.att_dim, self.num_heads, self.drop_prob, self.rt_mode)
            self.RTransformer_layers.append(h2h)
    
    
    def forward(self, rfs, rel_labels):
        
        batch_num = rfs.shape[0]
        if self.entity_drop_prob > 0:
            rfs = self.e_dropout(rfs)
        rfs = F.normalize(rfs, p=1, dim=-1)
            
        ht_embs = torch.matmul(rfs, self.r_ent_emb.weight)
        r_embs = self.rel_emb(rel_labels).unsqueeze(1)
        htr_embs = torch.cat((ht_embs, r_embs), dim=1)
        
        for layer in self.RTransformer_layers:
            htr_embs = layer(htr_embs)
        out_embs = torch.reshape(htr_embs, (batch_num, -1))
        out_scores = self.fcn(out_embs)
        
        return out_scores
    

class EncoderLayer(nn.Module):
    
    def __init__(self, emb_dim, att_dim, num_heads, drop_prob=0.1, mode='sRT'):
        super(EncoderLayer, self).__init__()
        
        self.emb_dim = emb_dim
        self.att_dim = att_dim
        self.num_heads = num_heads
        self.mode = mode
        
        self.multihead_att = Relational_MultiHeadAtten(emb_dim, att_dim, num_heads)
        self.LayerNorm1 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(drop_prob)
        
        self.ffn = FeedForward(emb_dim, att_dim, drop_prob)
        self.LayerNorm2 = nn.LayerNorm(emb_dim)
        self.dropout2 = nn.Dropout(drop_prob)
    
    
    def forward(self, hrt_emb):
        # self att
        att_embs = self.multihead_att(hrt_emb)
        att_embs = self.dropout1(att_embs)
        if self.mode == 'sRT':
            return att_embs
        # add and norm
        an_embs = self.LayerNorm1(att_embs + hrt_emb)
        # ffn
        ffn_embs = self.ffn(an_embs)
        ffn_embs = self.dropout2(ffn_embs)
        # add and norm
        out_embs = self.LayerNorm2(ffn_embs + an_embs)
        return out_embs
        
    
class Relational_MultiHeadAtten(nn.Module):
    
    def __init__(self, emb_dim, att_dim, num_heads):
        super(Relational_MultiHeadAtten, self).__init__()
        
        self.emb_dim = emb_dim
        self.att_dim = att_dim
        self.num_heads = num_heads
        
        self.multi_head_Q = nn.Parameter(torch.Tensor(num_heads, emb_dim, att_dim))
        self.multi_head_K = nn.Parameter(torch.Tensor(num_heads, emb_dim, att_dim))
        self.multi_head_V = nn.Parameter(torch.Tensor(num_heads, emb_dim, att_dim))
        self.multi_head_W = nn.Parameter(torch.Tensor(num_heads * att_dim, emb_dim))
        
        nn.init.xavier_uniform_(self.multi_head_Q, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.multi_head_K, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.multi_head_V, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.multi_head_W, gain=nn.init.calculate_gain('relu'))
        
    
    def forward(self, embs):
        # print('mutli_head_input_embs', embs.shape)
        input_embs = embs.unsqueeze(1)
        batch_num = embs.shape[0]
        elements_num = embs.shape[1]
        # print('input_embs', input_embs.shape)
        
        Q_weights = self.multi_head_Q.unsqueeze(0)
        K_weights = self.multi_head_K.unsqueeze(0)
        V_weights = self.multi_head_V.unsqueeze(0)
        # print('Q_weights', Q_weights.shape)

        Q_embs = torch.matmul(input_embs, Q_weights)
        K_embs = torch.matmul(input_embs, K_weights)
        V_embs = torch.matmul(input_embs, V_weights)
        
        self_atts = torch.softmax(torch.matmul(Q_embs, K_embs.permute(0, 1, 3, 2)) / math.sqrt(self.att_dim), dim=-1)
        
        att_embs = torch.reshape(torch.matmul(self_atts, V_embs).permute(0, 2, 1, 3), (batch_num, elements_num, -1))
        Z_embs = torch.matmul(att_embs, self.multi_head_W)
        
        return Z_embs


class FeedForward(nn.Module):
    
    def __init__(self, inp_dim, emb_dim, drop_prob=0.1):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(inp_dim, emb_dim)
        self.linear2 = nn.Linear(emb_dim, inp_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
    
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x