import torch

from torch.utils.data import Dataset

from utils.graph_utils import triplets2dgl_G


class TripletDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, triplets, entity2id, relation2id, params):
        self.triplets = triplets
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.params = params
        
        self.num_ent = len(entity2id)
        self.num_rel = len(relation2id)
        self.num_triplets = len(triplets)
    
    
    def __len__(self):
        return self.num_triplets
    
    
    def __getitem__(self, index):
        return self.triplets[index]