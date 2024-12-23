import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import DataLoader

from utils.dgl_utils import collate_dgl, move_batch_to_device_dgl


class Evaluator():
    def __init__(self, params, model, data):
        self.params = params
        self.model = model
        self.data = data
        self.batch_size = params.eval_batch_size

    def eval(self, testing=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        dataloader = DataLoader(self.data, batch_size=self.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=collate_dgl)

        self.model.eval()
        with torch.no_grad():
            pos_scores = torch.empty(0, device=self.params.device)
            neg_scores = torch.empty(0, device=self.params.device)
            pos_labels = torch.empty(0, device=self.params.device)
            neg_labels = torch.empty(0, device=self.params.device)
            all_ranks = torch.empty(0, device=self.params.device)
            
            for batch in tqdm(dataloader, desc='Evaluating'):
                data_pos, targets_pos, data_neg, targets_neg = move_batch_to_device_dgl(batch, self.params.device)
                if self.model.params.num_RT_layers > 0:
                    # R_scores_pos, G_scores_pos, = self.model(data_pos)
                    # R_scores_neg, G_scores_neg = self.model(data_neg)
                    # batch_num = R_scores_pos.shape[0]
                    # R_scores_all = torch.cat((R_scores_pos, R_scores_neg.view(batch_num, -1)), dim=1)
                    # G_scores_all = torch.cat((G_scores_pos, G_scores_neg.view(batch_num, -1)), dim=1)
                    # R_norm_scores_all = F.normalize(R_scores_all, p=1, dim=-1)
                    # G_norm_scores_all = F.normalize(G_scores_all, p=1, dim=-1)
                    # A_scores_all = R_norm_scores_all + G_norm_scores_all
                    # score_pos = A_scores_all[:, 0]
                    # score_neg = torch.flatten(A_scores_all[:, 1: ])
                    _, _, score_pos = self.model(data_pos)
                    _, _, score_neg = self.model(data_neg)
                else:
                    score_pos = self.model(data_pos)
                    score_neg = self.model(data_neg)
                # print(score_pos.shape)
                # print(score_neg.shape)
                pos_scores = torch.cat((pos_scores, score_pos))
                neg_scores = torch.cat((neg_scores, score_neg))
                pos_labels = torch.cat((pos_labels, targets_pos))
                neg_labels = torch.cat((neg_labels, targets_neg))
                
                if testing:
                    scores = torch.cat((score_pos, score_neg.view(score_pos.shape[0], -1)), dim=1)
                    ranks = torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1)[:, 0] + 1
                    all_ranks = torch.cat((all_ranks, ranks))

            # acc = metrics.accuracy_score(labels, preds)
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
