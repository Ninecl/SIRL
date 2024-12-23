import os
import time
import torch
import logging

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn import metrics
from torch.utils.data import DataLoader

from utils.dgl_utils import collate_dgl, move_batch_to_device_dgl


class Trainer():
    def __init__(self, params, model, train, valid_evaluator=None):
        self.model = model
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train
        self.num_rels = train.num_rels
        self.num_negs = params.num_neg_samples
        self.batch_size = params.batch_size

        self.updates_counter = 0

        model_params = list(self.model.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)

        self.G_criterion = nn.MarginRankingLoss(self.params.G_margin, reduction='sum')
        if self.params.num_RT_layers > 0:
            self.R_criterion = nn.MarginRankingLoss(self.params.R_margin, reduction='sum')
            self.A_criterion = nn.MarginRankingLoss(self.params.A_margin, reduction='sum')

        self.reset_training_state()


    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0


    def train_epoch(self):
        total_loss = 0

        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, collate_fn=collate_dgl)
        self.model.train()
        model_params = list(self.model.parameters())
        for b_idx, batch in enumerate(dataloader):
            data_pos, targets_pos, data_neg, targets_neg = move_batch_to_device_dgl(batch, self.params.device)
            self.optimizer.zero_grad()
            if self.params.num_RT_layers > 0:
                R_scores_pos, G_scores_pos, A_scores_pos = self.model(data_pos)
                R_scores_neg, G_scores_neg, A_scores_neg = self.model(data_neg)
                # R_scores_all = torch.cat((R_scores_pos, R_scores_neg.view(-1, self.num_negs)), dim=1)
                # G_scores_all = torch.cat((G_scores_pos, G_scores_neg.view(-1, self.num_negs)), dim=1)
                # Get R_loss
                R_scores_pos = torch.flatten(R_scores_pos.repeat((1, self.num_negs)))
                R_scores_neg = torch.flatten(R_scores_neg)
                R_loss = self.R_criterion(R_scores_pos, R_scores_neg, torch.ones_like(R_scores_pos).to(device=self.params.device))
                # Get G_loss
                G_scores_pos = torch.flatten(G_scores_pos.repeat((1, self.num_negs)))
                G_scores_neg = torch.flatten(G_scores_neg)
                G_loss = self.G_criterion(G_scores_pos, G_scores_neg, torch.ones_like(G_scores_pos).to(device=self.params.device))
                # Get A_loss
                # R_norm_scores_all = F.normalize(R_scores_all, p=1, dim=-1)
                # G_norm_scores_all = F.normalize(G_scores_all, p=1, dim=-1)
                # A_scores_all = R_norm_scores_all + G_norm_scores_all
                # A_scores_pos, A_scores_neg = A_scores_all[:, 0], A_scores_all[:, 1: ]
                # A_scores_pos = torch.flatten(A_scores_pos.unsqueeze(1).repeat((1, self.num_negs)))
                # A_scores_neg = torch.flatten(A_scores_neg)
                A_scores_pos = torch.flatten(A_scores_pos.repeat((1, self.num_negs)))
                A_scores_neg = torch.flatten(A_scores_neg)
                A_loss = self.A_criterion(A_scores_pos, A_scores_neg, torch.ones_like(A_scores_pos).to(device=self.params.device))
                # get loss
                loss = G_loss + R_loss + A_loss
            else:
                scores_pos = self.model(data_pos)
                scores_neg = self.model(data_neg)
                scores_pos = torch.flatten(scores_pos.repeat((1, self.num_negs)))
                scores_neg = torch.flatten(scores_neg)
                loss = self.G_criterion(scores_pos, scores_neg, torch.ones_like(scores_pos).to(device=self.params.device))
                
            loss.backward()
            self.optimizer.step()
            self.updates_counter += 1

            with torch.no_grad():
                total_loss += loss

        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        return total_loss, weight_norm


    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            loss, weight_norm = self.train_epoch()
            time_elapsed = time.time() - time_start
            logging.info('Epoch {:d} with loss: {:.3f}, best validation AUC: {:.3f}, weight_norm: {:.3f}, time cost: {:.3f}(s)'.format(epoch, loss, self.best_metric, weight_norm, time_elapsed))

            if self.valid_evaluator and epoch % self.params.eval_every == 0:
                time_start = time.time()
                result = self.valid_evaluator.eval()
                time_elapsed = time.time() - time_start
                logging.info('Valid Performance with AUC: {:.3f}, time cost: {:.3f}'.format(result['auc'], time_elapsed))
            
                if result['auc'] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result['auc']
                    self.not_improved_count = 0

                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result['auc']

            if epoch % self.params.save_every == 0:
                torch.save(self.model, os.path.join(self.params.exp_dir, 'last_model.pth'))


    def save_classifier(self):
        torch.save(self.model, os.path.join(self.params.exp_dir, 'best_model.pth'))  # Does it overwrite or fuck with the existing file?
        logging.info('Better models found w.r.t accuracy. Saved it!')
