import abc
import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from .RTransformer import EncoderLayer


class Aggregator(nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, node):
        curr_emb = node.mailbox['curr_emb'][:, 0, :]  # (B, F)
        nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), node.mailbox['msg']).squeeze(1)  # (B, F)
        # nei_msg, _ = torch.max(node.mailbox['msg'], 1)  # (B, F)

        new_emb = self.update_embedding(curr_emb, nei_msg)

        return {'h': new_emb}

    @abc.abstractmethod
    def update_embedding(curr_emb, nei_msg):
        pass


class SumAggregator(Aggregator):
    def __init__(self):
        super(SumAggregator, self).__init__()

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = nei_msg + curr_emb

        return new_emb


class MLPAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(MLPAggregator, self).__init__()
        self.linear = nn.Linear(2 * emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        inp = torch.cat((nei_msg, curr_emb), 1)
        new_emb = F.relu(self.linear(inp))

        return new_emb


class GRUAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(GRUAggregator, self).__init__()
        self.gru = nn.GRUCell(emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = self.gru(nei_msg, curr_emb)

        return new_emb


class Transpooling(nn.Module):
    
    def __init__(self, num_rels, inp_dim, emb_dim, att_dim, num_heads):
        super(Transpooling, self).__init__()
        
        self.inp_dim = inp_dim
        self.emb_dim = emb_dim
        self.att_dim = att_dim
        
        self.self_loop_idx = num_rels
        self.self_loop_weight = nn.Parameter(torch.Tensor(inp_dim, emb_dim))
        
        self.relational_Q = nn.Parameter(torch.Tensor(num_rels + 1, emb_dim, att_dim))
        self.relational_K = nn.Parameter(torch.Tensor(num_rels + 1, emb_dim, att_dim))
        self.relational_V = nn.Parameter(torch.Tensor(num_rels + 1, emb_dim, att_dim))
        
        self.ffn = nn.Linear(att_dim * num_heads, emb_dim)
        
        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.relational_Q, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.relational_K, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.relational_V, gain=nn.init.calculate_gain('relu'))
    
    
    def forward(self, node):
        curr_emb = torch.matmul(node.data['h'], self.self_loop_weight)
        relation_idxs = node.mailbox['msg_type']
        batch_num = curr_emb.shape[0]
        
        # print('msg', node.mailbox['msg'].shape)
        input_embs = torch.cat((curr_emb.unsqueeze(1), node.mailbox['msg']), dim=1).view(-1, 1, self.emb_dim)
        # print('input embs', input_embs.shape)
        
        qv_idxs = torch.cat((node.data['r_label'].unsqueeze(1), node.mailbox['r_label']), dim=1)
        # print('qv_idxs', qv_idxs.shape)
        self_loop_idxs = torch.tensor([self.self_loop_idx, ], device=curr_emb.device).tile(batch_num).unsqueeze(1)
        k_idxs = torch.cat((self_loop_idxs, relation_idxs), dim=1)
        # print('k_idxs', k_idxs.shape)
        
        Q_weights = torch.index_select(self.relational_Q, 0, qv_idxs.view(-1))
        # print('Q_weights', Q_weights.shape)
        K_weights = torch.index_select(self.relational_K, 0, k_idxs.view(-1))
        # print('K_weights', K_weights.shape)
        V_weights = torch.index_select(self.relational_V, 0, qv_idxs.view(-1))
        # print('V_weights', V_weights.shape)
        
        Q_embs = torch.bmm(input_embs, Q_weights).view(batch_num, -1, self.att_dim)
        K_embs = torch.bmm(input_embs, K_weights).view(batch_num, -1, self.att_dim)
        V_embs = torch.bmm(input_embs, V_weights).view(batch_num, -1, self.att_dim)
        
        # print('Q_embs', Q_embs.shape)
        # print('K_embs', K_embs.shape)
        # print('V_embs', V_embs.shape)
        
        self_atts = torch.softmax(torch.bmm(Q_embs, K_embs.permute(0, 2, 1)) / math.sqrt(self.att_dim), dim=1)
        # print('self_atts', self_atts.shape)
        out_embs = self.ffn(torch.bmm(self_atts, V_embs)[:, 0])
        # print('out_embs', out_embs.shape)
        
        return {'h': out_embs}
    
    
class Graphpooling(nn.Module):
    
    def __init__(self, emb_dim, att_dim, num_heads, num_gcn_layers):
        super(Graphpooling, self).__init__()
        
        self.inp_dim = num_gcn_layers * emb_dim
        self.att_dim = att_dim
        self.num_heads = num_heads
        
        self.graphpooling = EncoderLayer(self.inp_dim, self.att_dim, self.num_heads)
        
    
    def forward(self, g):
        input_embs = g.ndata['repr'].view(-1, self.inp_dim).unsqueeze(0)
        graph_embs = self.graphpooling(input_embs).squeeze()
        # print('graphpooling embs', graph_embs.shape)
        out_embs = torch.sum(graph_embs, dim=0)
        # print('graphpooling out embs', out_embs.shape)
        
        return out_embs
    

class Simple_Graphpooling(nn.Module):
    
    def __init__(self, emb_dim, att_dim, num_heads, num_gcn_layers):
        super(Simple_Graphpooling, self).__init__()
        
        self.inp_dim = num_gcn_layers * emb_dim
        self.out_dim = num_gcn_layers * emb_dim
        self.att_dim = att_dim
        self.num_heads = num_heads
        
        self.multi_head_Q = nn.Parameter(torch.Tensor(self.num_heads, self.inp_dim, self.att_dim))
        self.multi_head_K = nn.Parameter(torch.Tensor(self.num_heads, self.inp_dim, self.att_dim))
        self.multi_head_V = nn.Parameter(torch.Tensor(self.num_heads, self.inp_dim, self.att_dim))
        
        self.ffn = nn.Linear(self.att_dim * self.num_heads, self.out_dim)
        
        nn.init.xavier_uniform_(self.multi_head_Q, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.multi_head_K, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.multi_head_V, gain=nn.init.calculate_gain('relu'))
    
    
    def forward(self, g):
        input_embs = g.ndata['repr'].view(-1, self.inp_dim)

        Q_embs = torch.matmul(input_embs, self.multi_head_Q)
        K_embs = torch.matmul(input_embs, self.multi_head_K)
        V_embs = torch.matmul(input_embs, self.multi_head_V)
        
        self_atts = torch.softmax(torch.bmm(Q_embs, K_embs.permute(0, 2, 1)) / math.sqrt(self.att_dim), dim=-1)
        att_embs = torch.reshape(torch.bmm(self_atts, V_embs).permute(1, 0, 2), (-1, self.att_dim * self.num_heads))
        out_embs = torch.sum(self.ffn(att_embs), dim=0)
        
        return out_embs