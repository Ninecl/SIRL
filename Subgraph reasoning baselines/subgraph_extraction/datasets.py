import os
import dgl
import lmdb
import json
import torch
import logging

import numpy as np

from torch.utils.data import Dataset

from subgraph_extraction.graph_sampler import sample_neg, links2subgraphs
from utils.data_utils import read_triplets, read_se_support_triplets, process_triplets, load_json_file, dump_json_file, deserialize
from utils.graph_utils import ssp_multigraph_to_dgl


def generate_subgraph_datasets(params, testing=False, query_mode='trans'):
    # save_path
    saved_relation2id_path = os.path.join(params.main_dir, f'data/{params.dataset}/relation2id.json')
    if not testing:
        triplets_paths = params.file_paths['ori']
        saved_entity2id_path = os.path.join(params.main_dir, f'data/{params.dataset}/ori/entity2id.json')
        params.db_path = os.path.join(params.main_dir, f'data/{params.dataset}/ori/subgraphs_neg{params.num_neg_samples}_hop{params.hop}')
        # load triplets
        support_triplets = read_triplets(triplets_paths['train'], params.triplets_type)
        query_triplets = read_triplets(triplets_paths['valid'], params.triplets_type)
    else:
        ori_triplets_paths = params.file_paths['ori']
        emg_triplets_paths = params.file_paths['emg']
        # load triplets
        ori_support_triplets = read_triplets(ori_triplets_paths['train'], params.triplets_type)
        emg_support_triplets = read_triplets(emg_triplets_paths['support'], params.triplets_type)
        if query_mode == 'trans':
            params.db_path = os.path.join(params.main_dir, f'data/{params.dataset}/ori/subgraphs_neg{params.num_neg_samples}_hop{params.hop}_trans')
            saved_entity2id_path = os.path.join(params.main_dir, f'data/{params.dataset}/ori/entity2id.json')
            support_triplets = ori_support_triplets
            query_triplets = read_triplets(ori_triplets_paths['test'], params.triplets_type)
        elif query_mode == 'ind':
            params.db_path = os.path.join(params.main_dir, f'data/{params.dataset}/emg/subgraphs_neg{params.num_neg_samples}_hop{params.hop}_ind')
            saved_entity2id_path = os.path.join(params.main_dir, f'data/{params.dataset}/emg/entity2id.json')
            support_triplets = ori_support_triplets + emg_support_triplets
            query_triplets = read_triplets(emg_triplets_paths['query'], params.triplets_type)
        elif query_mode == 'IT':
            params.db_path = os.path.join(params.main_dir, f'data/{params.dataset}/ori/subgraphs_neg{params.num_neg_samples}_hop{params.hop}_IT')
            saved_entity2id_path = os.path.join(params.main_dir, f'data/{params.dataset}/ori/entity2id.json')
            support_triplets = ori_support_triplets + emg_support_triplets
            query_triplets = read_triplets(ori_triplets_paths['test'], params.triplets_type)
    
    if params.max_nodes_per_hop is not None:
        params.db_path += f"_max{params.max_nodes_per_hop}"
    
    # construct graphs
    if testing:
        logging.info(f"{len(support_triplets)} support triplets in emg graph")
        relation2id = load_json_file(saved_relation2id_path)
        adj_list, rf_list, triplets2id_data, entity2id, relation2id = process_triplets(support_triplets, query_triplets, relation2id)
    else:
        if os.path.exists(saved_relation2id_path) and os.path.exists(params.db_path):
            relation2id = load_json_file(saved_relation2id_path)
            adj_list, rf_list, triplets2id_data, entity2id, relation2id = process_triplets(support_triplets, query_triplets, relation2id)
        else:
            adj_list, rf_list, triplets2id_data, entity2id, relation2id = process_triplets(support_triplets, query_triplets)

    if not os.path.exists(params.db_path) or params.resample:
        logging.info(f"Extracting subgraphs DB to {params.db_path}")
        graphs = {}
        for triplets_type in ['support', 'query']:
            if testing and triplets_type == 'support':
                continue
            graphs[triplets_type] = {'pos': triplets2id_data[triplets_type]}
            logging.info("Sampling {} negative link(s) for {}.".format(params.num_neg_samples, triplets_type))
            graphs[triplets_type]['neg'] = sample_neg(adj_list, graphs[triplets_type]['pos'], params.num_neg_samples)
        
        # extract subgraphs
        links2subgraphs(adj_list, rf_list, graphs, params)
        dump_json_file(saved_entity2id_path, entity2id)
        if not testing:
            dump_json_file(saved_relation2id_path, relation2id)
    else:
        logging.info(f"Loading subgraphs DB from {params.db_path}")
    
    return adj_list, triplets2id_data, entity2id, relation2id


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, db_name_pos, db_name_neg, params, A_list):

        self.main_env = lmdb.open(params.db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.num_neg_samples = params.num_neg_samples
        self.max_n_label = np.array([params.hop, params.hop])

        ssp_graph = A_list
        self.num_rels = len(ssp_graph)

        # Add transpose matrices to handle both directions of relations.
        if params.add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.ssp_graph = ssp_graph
        
        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
            self.avg_subgraph_nodes_pos = int.from_bytes(txn.get('avg_subgraph_nodes'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
            self.avg_subgraph_nodes_neg = int.from_bytes(txn.get('avg_subgraph_nodes'.encode()), byteorder='little')
            

        logging.info(f"{self.num_graphs_pos} triplets in {db_name_pos}, {self.num_graphs_neg} in {db_name_neg}")
        logging.info(f"{self.avg_subgraph_nodes_pos} node(s) per subgraph in {db_name_pos}, {self.avg_subgraph_nodes_neg} node(s) per subgraph in {db_name_neg}")


    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            nodes_pos, r_label_pos, rf_pos, n_labels_pos = deserialize(txn.get(str_id))
            subgraph_pos = self.prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)
        subgraphs_neg = []
        r_labels_neg = []
        rfs_neg = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                nodes_neg, r_label_neg, rf_neg, n_labels_neg = deserialize(txn.get(str_id))
                subgraphs_neg.append(self.prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg))
                r_labels_neg.append(r_label_neg)
                rfs_neg.append(rf_neg)

        return subgraph_pos, r_label_pos, rf_pos, subgraphs_neg, r_labels_neg, rfs_neg


    def __len__(self):
        return self.num_graphs_pos

    
    def prepare_subgraphs(self, nodes, r_label, n_labels):
        subgraph = self.graph.subgraph(nodes)
        subgraph.edata['type'] = self.graph.edata['type'][subgraph.edata[dgl.EID]]
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)
        
        if subgraph.has_edges_between(0, 1):
            edges_btw_roots = subgraph.edge_ids(0, 1)
        else:
            edges_btw_roots = torch.tensor([], dtype=torch.int64)
            
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        if rel_link.squeeze().nelement() == 0:
            subgraph.add_edges(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            
        subgraph = self.prepare_features(subgraph, n_labels, r_label)
        
        return subgraph
    
    
    def prepare_features(self, subgraph, n_labels, r_label):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        subgraph.ndata['h'] = torch.FloatTensor(label_feats)

        n_ids = np.zeros(n_nodes)
        n_ids[0] = 1  # head
        n_ids[1] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)
        subgraph.ndata['r_label'] = torch.tensor(r_label * np.ones(n_nodes), dtype=torch.long)
        
        return subgraph