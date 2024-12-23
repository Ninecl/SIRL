import os
import torch
import logging
import argparse

from warnings import simplefilter
from scipy.sparse import SparseEfficiencyWarning

from manager.trainer import Trainer
from manager.evaluator import Evaluator
from utils.data_utils import read_triplets2id
from utils.initialization_utils import initialize_experiment, initialize_model
from dataload.GNNs_dataloader import TripletDataset


def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)
    
    train_triplets, entity2id, relation2id = read_triplets2id(params.file_paths['train'], 'hrt', with_head=False, allow_emerging=True)
    valid_triplets, entity2id, relation2id = read_triplets2id(params.file_paths['valid'], 'hrt', entity2id, relation2id, with_head=False, allow_emerging=True)
    
    train_dataset = TripletDataset(train_triplets, entity2id, relation2id, params)
    valid_dataset = TripletDataset(valid_triplets, entity2id, relation2id, params)
    params.num_ent = train_dataset.num_ent
    params.num_rel = train_dataset.num_rel
    params.entity2id = entity2id
    params.relation2id = relation2id
    
    if params.model_type == "MEAN":
        from model.GNNs_based.MEAN import MEAN as dgl_model
    else:
        assert RuntimeError("Wrong model name")
        
    model = initialize_model(params, dgl_model, params.load_model)
    
    logging.info(f"Device: {params.device}")
    logging.info(f"Input dim : {params.emb_dim}, # Relations : {params.num_rel}")
    valid_evaluator = Evaluator(params, model, train_dataset, valid_dataset)
    trainer = Trainer(params, model, train_dataset, valid_evaluator)
    
    logging.info('Starting training with full batch...')
    trainer.train()
    
    
if __name__ == '__main__':
    
    log_format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)

    parser = argparse.ArgumentParser(description='MEAN model')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="MEAN",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str, choices=['FB15k-237', 'NELL-995'], 
                        help="Dataset string")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--load_model', action='store_true',
                        help='Load existing model?')
    parser.add_argument("--train_file", "-tf", type=str, default="support",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="query",
                        help="Name of file containing validation triplets")
    parser.add_argument("--triplets_type", "-tt", type=str, choices=['htr', 'hrt'], default="hrt",
                        help="The triplets form in files")

    # Training regime params
    parser.add_argument("--model_type", "-mt", type=str, default='MEAN', choices=['MEAN', 'LAN'], 
                        help="Which model to use")
    parser.add_argument("--num_epochs", "-ne", type=int, default=2000,
                        help="number of epoches for training")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Interval of epochs to evaluate the model?")
    parser.add_argument("--save_every", type=int, default=2,
                        help="Interval of epochs to save a checkpoint of the model?")
    parser.add_argument("--early_stop", type=int, default=100,
                        help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate of the optimizer")
    parser.add_argument("--clip", type=int, default=1000,
                        help="Maximum gradient norm allowed")
    parser.add_argument("--l2", type=float, default=5e-4,
                        help="Regularization constant for GNN weights")
    parser.add_argument("--margin", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")

    # Data processing pipeline params
    parser.add_argument("--max_train_links", type=int, default=1000000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--batch_size", "-bs", type=int, default=2000,
                        help="Batch size")
    parser.add_argument("--eval_batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--num_neg_samples", '-neg', type=int, default=8,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--num_workers", "-nw", type=int, default=8,
                        help="Number of dataloading processes")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations')

    # Model params
    # overall
    parser.add_argument("--emb_dim", "-dim", type=int, default=32,
                        help="Entity embedding size")
    # GNN
    parser.add_argument("--num_gnn_layers", "-l", type=int, default=3,
                        help="Number of GCN layers")
    parser.add_argument("--num_bases", "-b", type=int, default=4,
                        help="Number of basis functions to use for GCN weights")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout rate of embeddings")
    parser.add_argument("--edge_dropout", type=float, default=0.5,
                        help="Dropout rate in edges of the subgraphs")
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mean', 'max'], default='mean',
                        help='what type of aggregation to do in gnn msg passing')
    parser.add_argument('--score_fusion', '-sfm', type=str, choices=['tanh', 'sigmoid', 'norm', 'identity'], default='identity',
                        help='how to fuse G and R scores')

    params = parser.parse_args()

    initialize_experiment(params, __file__)
    # set device
    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')
    
    # train/valid data path
    params.file_paths = {
        'train': os.path.join(params.main_dir, f'dataset/{params.dataset}/ori/train.txt'),
        'valid': os.path.join(params.main_dir, f'dataset/{params.dataset}/ori/valid.txt')
    }
    
    main(params)