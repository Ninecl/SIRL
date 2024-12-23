import os
import time
import tqdm
import torch
import logging
import argparse

from manager.evaluator import Evaluator
from utils.initialization_utils import initialize_model
from utils.data_utils import read_triplets2id
from dataload.GNNs_dataloader import TripletDataset

def main(params, model):
    
    # load model
    # load some parameters the same as training
    params.emb_dim = model.emb_dim
    params.add_traspose_rels = model.params.add_traspose_rels
    params.data_path = f'./dataset/{params.dataset}/{params.query_dataset}'
    # params.max_nodes_per_hop = model.params.max_nodes_per_hop if params.max_nodes_per_hop == -1 else params.max_nodes_per_hop
    
    ori_entity2id, ori_relation2id = model.entity2id, model.relation2id
    ori_train_triplets, ori_entity2id, ori_relation2id = read_triplets2id(params.file_paths['ori']['train'], 'hrt', ori_entity2id, ori_relation2id)
    support_triplets, entity2id, relation2id = read_triplets2id(params.file_paths[f'{params.query_dataset}']['support'], 'hrt', 
                                                                ori_entity2id, ori_relation2id, allow_emerging=True)
    query_triplets, entity2id, relation2id = read_triplets2id(params.file_paths[f'{params.query_dataset}']['query'], 'hrt', 
                                                                entity2id, relation2id)
    
    support_dataset = TripletDataset(ori_train_triplets + support_triplets, entity2id, relation2id, params)
    query_triplets = TripletDataset(query_triplets, entity2id, relation2id, params)
    
    params.num_ent = support_dataset.num_ent
    params.num_rel = support_dataset.num_rel
    
    test_evaluator = Evaluator(params, model, support_dataset, query_triplets)
    
    results = test_evaluator.eval(testing=True)

    return results['auc'], results['mrr'], results['hits_1'], results['hits_3'], results['hits_10']


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Testing script for hits@10')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="fb_v2_margin_loss",
                        help="Experiment name. Log file with this name will be created")
    parser.add_argument("--dataset", "-d", type=str, choices=['FB15k-237', 'NELL-995', 'CoDEX-M'], default="FB15k-237",
                        help="Path to dataset")
    parser.add_argument("--eval_batch_size", "-bs", type=int, default=50,
                        help="Evaluation batch size")
    parser.add_argument("--triplets_type", "-tt", type=str, choices=['htr', 'hrt'], default="hrt",
                        help="The triplets form in files")
    parser.add_argument("--query_dataset", "-qd", type=str, choices=['EQ', 'MO', 'LE'], default="LE",
                        help="query dataset")

    # Data process setup
    parser.add_argument('--se_support', '-ss', action='store_true',
                        help='Whether load support triplets in previous stage.')
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument('--use_cuda', '-uc', type=bool, default=True,
                        help='Whether use cuda.')
    parser.add_argument("--num_workers", '-nw', type=int, default=6,
                        help="Number of dataloading processes")
    parser.add_argument('--device', '-de', type=int, default=0, choices=[-1, 0, 1, 2, 3],
                        help='Which gpu to use.')
    parser.add_argument('--num_neg_samples', '-ns', type=int, default=500,
                        help='Number of negative sample for each link.')
    parser.add_argument('--resample', action='store_true', 
                        help='Whether resample negative links.')
    parser.add_argument('--only_extract', action='store_true', 
                        help='Only extract subgraphs without test.')

    params = parser.parse_args()

    # set path
    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))))
    params.exp_dir = os.path.join(params.main_dir, 'experiments', params.experiment_name)
    file_handler = logging.FileHandler(os.path.join(params.exp_dir, f'rank_test_{time.time()}.log'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    params.file_paths = {
        'ori': {'train': os.path.join(params.main_dir, f'dataset/{params.dataset}/ori/train.txt'),
                'valid': os.path.join(params.main_dir, f'dataset/{params.dataset}/ori/valid.txt')
                },
        'LE': {'support': os.path.join(params.main_dir, f'dataset/{params.dataset}/LE/train.txt'),
                'query': os.path.join(params.main_dir, f'dataset/{params.dataset}/LE/test.txt'),
                },
        'EQ': {'support': os.path.join(params.main_dir, f'dataset/{params.dataset}/EQ/train.txt'),
                'query': os.path.join(params.main_dir, f'dataset/{params.dataset}/EQ/test.txt'),
                },
        'MO': {'support': os.path.join(params.main_dir, f'dataset/{params.dataset}/MO/train.txt'),
                'query': os.path.join(params.main_dir, f'dataset/{params.dataset}/MO/test.txt'),
                }
    }

    # 设置gpu
    if params.use_cuda and torch.cuda.is_available() and params.device >= 0:
        params.device = torch.device('cuda:%d' % params.device)
    else:
        params.device = torch.device('cpu')

    # load model
    model = initialize_model(params, None, True)
    
    logger.info(f"Test for {params.query_dataset} KGC.")
    auc, mrr, hits_1, hits_3, hits_10 = main(params, model)
    logger.info('RESULT: AUC | MRR | Hits@1 | Hits@3 | Hits@10 : {:.5f} | {:.5f} | {:.5f} | {:.5f} | {:.5f}'.format(auc, mrr, hits_1, hits_3, hits_10))