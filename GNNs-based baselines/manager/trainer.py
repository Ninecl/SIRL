import os
import time
import torch
import logging

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from utils.data_utils import collate_function, sample_neg_triplets
from utils.graph_utils import triplets2dgl_G


class Trainer():
    def __init__(self, params, model, train, valid_evaluator=None):
        self.model = model
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train

        self.updates_counter = 0

        model_params = list(self.model.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')
        # self.collate_dgl = collate_dgl

        self.reset_training_state()


    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0


    def train_epoch(self):
        total_loss = 0
            
        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, collate_fn=collate_function)
        self.model.train()
        model_params = list(self.model.parameters())
        for b_idx, batch in enumerate(dataloader):
            dgl_g = triplets2dgl_G(batch, self.params.device)
            ent_embs = self.model(dgl_g)
            pos_triplets, neg_triplets = sample_neg_triplets(batch, self.params.num_ent, self.params.num_neg_samples, self.params.device)
            score_pos = self.model.score(pos_triplets, ent_embs)
            score_neg = self.model.score(neg_triplets, ent_embs)
            
            loss = self.criterion(score_pos, score_neg.view(len(score_pos), -1).mean(dim=1).view(score_pos.shape), torch.ones_like(score_pos).to(device=self.params.device))
            # print(score_pos, score_neg, loss)
            loss.backward()
            self.optimizer.step()
            self.updates_counter += 1
            self.model.ent_embs.weight.data = ent_embs

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
