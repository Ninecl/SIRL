import os
import torch
import logging
import argparse

from warnings import simplefilter
from scipy.sparse import SparseEfficiencyWarning

from managers.trainer import Trainer
from managers.evaluator import Evaluator
from utils.initialization_utils import initialize_experiment, initialize_model
from subgraph_extraction.datasets import generate_subgraph_datasets, SubgraphDataset


def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)
    
    adj_list, triplets2id_data, entity2id, relation2id = generate_subgraph_datasets(params)
    
    train_dataset = SubgraphDataset('support_pos', 'support_neg', params, adj_list)
    valid_dataset = SubgraphDataset('query_pos', 'query_neg', params, adj_list)
    
    params.num_rels = train_dataset.num_rels
    params.aug_num_rels = train_dataset.aug_num_rels
    params.max_n_label = train_dataset.max_n_label
    params.inp_dim = (params.hop + 1) * 2
    logging.info(f"Max distance from sub : {params.max_n_label[0]}, Max distance from obj : {params.max_n_label[1]}")
    
    if params.model_type == "Grail":
        from model.framework import Grail as dgl_model
        params.num_RT_layers = 0
    elif params.model_type == "ISE2":
        from model.framework import ISE2 as dgl_model
    else:
        assert RuntimeError("Wrong model name")
        
    model = initialize_model(params, dgl_model, params.load_model)
    
    logging.info(f"Device: {params.device}")
    logging.info(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")
    valid_evaluator = Evaluator(params, model, valid_dataset)
    trainer = Trainer(params, model, train_dataset, valid_evaluator)
    
    logging.info('Starting training with full batch...')
    trainer.train()
    
    
if __name__ == '__main__':
    
    log_format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)

    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str,
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
    parser.add_argument("--num_epochs", "-ne", type=int, default=100,
                        help="Learning rate of the optimizer")
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
    parser.add_argument("--R_margin", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")
    parser.add_argument("--G_margin", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")
    parser.add_argument("--A_margin", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")

    # Data processing pipeline params
    parser.add_argument("--max_train_links", type=int, default=1000000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=3,
                        help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument('--model_type', '-mt', type=str, choices=['Grail', 'ISE2'], default='Grail',
                        help='which model to use')
    parser.add_argument("--batch_size", "-bs", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--eval_batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--num_neg_samples", '-neg', type=int, default=1,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--num_workers", "-nw", type=int, default=8,
                        help="Number of dataloading processes")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')
    parser.add_argument('--resample', '-re', action='store_true',
                        help='whether resample and extract subgraphs')

    # Model params
    # overall
    parser.add_argument("--emb_dim", "-dim", type=int, default=32,
                        help="Entity embedding size")
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=32,
                        help="Relation embedding size for attention")
    parser.add_argument("--att_dim", "-ad", type=int, default=64,
                        help="Transpooling attention embedding size")
    parser.add_argument("--num_heads", "-nh", type=int, default=8,
                        help="Number of attention head in transpooling")
    # GNN
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=3,
                        help="Number of GCN layers")
    parser.add_argument("--num_bases", "-b", type=int, default=4,
                        help="Number of basis functions to use for GCN weights")
    parser.add_argument("--G_dropout", type=float, default=0,
                        help="Dropout rate in GNN layers")
    parser.add_argument("--edge_dropout", type=float, default=0.5,
                        help="Dropout rate in edges of the subgraphs")
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum',
                        help='what type of aggregation to do in gnn msg passing')
    # GP
    parser.add_argument('--graphpooling_mode', '-GP_m', type=str, choices=['sGP', 'GP'], default='sGP',
                        help='which graphpooling to use')
    # RT
    parser.add_argument("--relationalTransformer_mode", "-RT_m", type=str, choices=['sRT', 'RT'], default='sRT', 
                        help="which RT to use")
    parser.add_argument("--num_RT_layers", "-n_RT", type=int, default=1,
                        help="num of layers in RTransformer")
    parser.add_argument("--R_dropout", type=float, default=0.5,
                        help="Dropout rate in RTransformer")
    parser.add_argument("--entity_dropout", type=float, default=0.5,
                        help="Dropout rate in entities of the rfs")
    # socres
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True,
                        help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--has_attn', '-attn', type=bool, default=True,
                        help='whether to have attn in model or not')
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
        'ori': {'train': os.path.join(params.main_dir, f'data/{params.dataset}/ori/train.txt'),
                'valid': os.path.join(params.main_dir, f'data/{params.dataset}/ori/valid.txt'),
                'test': os.path.join(params.main_dir, f'data/{params.dataset}/ori/test.txt')
                },
        'emg': {'train': os.path.join(params.main_dir, f'data/{params.dataset}/emg/train.txt'),
                'test': os.path.join(params.main_dir, f'data/{params.dataset}/emg/valid.txt'),
                }
    }
    
    main(params)