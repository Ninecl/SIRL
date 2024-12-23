import os
import torch
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import DataLoader

from utils.data_utils import write_tensor_scores, collate_function, sample_neg_triplets
from utils.graph_utils import triplets2dgl_G


class Evaluator():
    def __init__(self, params, model, support, query):
        self.params = params
        self.model = model
        self.support = support
        self.query = query
        self.batch_size = params.eval_batch_size

    def eval(self, testing=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        dataloader = DataLoader(self.query, batch_size=self.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=collate_function)

        self.model.eval()
            
        with torch.no_grad():
            pos_scores = torch.empty(0, device=self.params.device)
            neg_scores = torch.empty(0, device=self.params.device)
            pos_labels = torch.empty(0, device=self.params.device)
            neg_labels = torch.empty(0, device=self.params.device)
            all_ranks = torch.empty(0, device=self.params.device)
            
            if not testing:
                ent_embs = self.model.ent_embs.weight
            else:
                support_triplets = self.support.triplets
                dgl_G = triplets2dgl_G(support_triplets, self.params.device)
                ori_ent_embs = self.model.ent_embs.weight
                self.model.ent_embs = nn.Embedding(self.params.num_ent, self.params.emb_dim, device=self.params.device)
                ori_embs_idx = torch.arange(0, len(ori_ent_embs), device=self.params.device)
                self.model.ent_embs.weight[ori_embs_idx] = ori_ent_embs
                ent_embs = self.model(dgl_G)
                
            
            for batch in tqdm(dataloader, desc='Evaluating'):
                pos_triplets, neg_triplets = sample_neg_triplets(batch, self.params.num_ent, self.params.num_neg_samples, self.params.device)
                score_pos = self.model.score(pos_triplets, ent_embs)
                score_neg = self.model.score(neg_triplets, ent_embs)
                pos_scores = torch.cat((pos_scores, score_pos))
                neg_scores = torch.cat((neg_scores, score_neg))
                pos_labels = torch.cat((pos_labels, torch.ones_like(score_pos)))
                neg_labels = torch.cat((neg_labels, torch.zeros_like(score_neg)))
                
                if testing:
                    score_pos = score_pos.unsqueeze(1)
                    scores = torch.cat((score_pos, score_neg.view(score_pos.shape[0], -1)), dim=1)
                    ranks = torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1)[:, 0] + 1
                    all_ranks = torch.cat((all_ranks, ranks))
            
            if testing:
                pos_scores_path = os.path.join(self.params.data_path, 'pos_scores.txt')
                write_tensor_scores(pos_scores_path, pos_scores)
                neg_scores_path = os.path.join(self.params.data_path, 'neg_scores.txt')
                write_tensor_scores(neg_scores_path, neg_scores)
            
            labels = torch.cat((pos_labels, neg_labels)).cpu()
            scores = torch.cat((pos_scores, neg_scores)).cpu()
            auc = metrics.roc_auc_score(labels, scores)

            if testing:
                isHit1List = [x for x in all_ranks if x <= 1]
                isHit5List = [x for x in all_ranks if x <= 3]
                isHit10List = [x for x in all_ranks if x <= 10]
                hits_1 = len(isHit1List) / len(all_ranks)
                hits_3 = len(isHit5List) / len(all_ranks)
                hits_10 = len(isHit10List) / len(all_ranks)
                mrr = torch.mean(1 / all_ranks)
                
                return {'auc': auc, 'mrr': mrr, 'hits_1': hits_1, 'hits_3': hits_3, 'hits_10': hits_10}
        
            return {'auc': auc}
