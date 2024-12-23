import dgl
import torch

def triplets2dgl_G(triplets, device=None):
    
    triplets = torch.tensor(triplets)
    h, r, t = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    ht = torch.concatenate((h, t))
    
    entity_ori_ID, ht_dgl_ID = torch.unique(ht, return_inverse=True)
    h_dgl, t_dgl = ht_dgl_ID.view(2, -1)
    
    dgl_G = dgl.graph((h_dgl, t_dgl))
    dgl_G.ndata['id'] = entity_ori_ID
    dgl_G.edata['type'] = r
    
    if device is not None:
        dgl_G = dgl_G.to(device)
    
    return dgl_G