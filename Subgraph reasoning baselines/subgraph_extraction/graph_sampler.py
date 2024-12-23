import dgl
import time
import lmdb
import torch
import logging

import numpy as np
import scipy.sparse as ssp
import multiprocessing as mp

from tqdm import tqdm

from utils.data_utils import serialize
from utils.graph_utils import incidence_matrix, matrix_to_graph, remove_nodes


def sample_neg(adj_list, pos_edges, num_neg_samples=1):
    num_pos_edges = len(pos_edges)
    neg_edges = []

    # sample negative links for train/test
    n, r = adj_list[0].shape[0], len(adj_list)
    num_total_neg = num_neg_samples * num_pos_edges

    pbar = tqdm(total=num_total_neg, desc=f'Sampling')
    while len(neg_edges) < num_total_neg:
        neg_head, neg_tail, rel = pos_edges[pbar.n % num_pos_edges]
        
        if np.random.uniform() < 0.5:
            neg_head = np.random.choice(n)
        else:
            neg_tail = np.random.choice(n)
                
        if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
            neg_edges.append([neg_head, neg_tail, rel])
            pbar.update(1)
    pbar.close()

    neg_edges = np.array(neg_edges)
    return neg_edges


def links2subgraphs(A_list, rf_list, graphs, params, testing=False):
    A = incidence_matrix(A_list)
    uA = A + A.T
    uGraph = matrix_to_graph(uA)
    intialize_worker(A, rf_list, uGraph, params)
    
    BYTES_PER_DATUM = get_average_subgraph_size(2000, list(graphs.values())[-1]['pos'], A, rf_list, uGraph, params) * 2
    if not testing:
        total_link_num = 0
        for triplets_type, triplets_data in graphs.items():
            total_link_num += (len(triplets_data['pos']) + len(triplets_data['neg']))
        map_size = total_link_num * BYTES_PER_DATUM
        env = lmdb.open(params.db_path, map_size=map_size, max_dbs=6)
    else:
        total_link_num = len(graphs['query']['pos']) + len(graphs['query']['neg'])
        map_size = total_link_num * BYTES_PER_DATUM
        env = lmdb.open(params.db_path, map_size=map_size, max_dbs=3)
    
    for triplets_type, triplets_data in graphs.items():
        logging.info(f"Extracting enclosing subgraphs for positive links in {triplets_type} set")
        db_name_pos = triplets_type + '_pos'
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(triplets_data['pos'], env, split_env)
        
        logging.info(f"Extracting enclosing subgraphs for negative links in {triplets_type} set")
        db_name_neg = triplets_type + '_neg'
        split_env = env.open_db(db_name_neg.encode())
        extraction_helper(triplets_data['neg'], env, split_env)


def extraction_helper(links, env, split_env):

    with mp.Pool(processes=None) as p:
        args_ = zip(range(len(links)), links)
        total_num_nodes = 0
        for (str_id, datum) in tqdm(p.imap(extract_save_subgraph, args_), total=len(links), desc='Extracting'):
            total_num_nodes += len(datum['nodes'])
            with env.begin(write=True, db=split_env) as txn:
                txn.put(str_id, serialize(datum))
        avg_subgraph_nodes = int(total_num_nodes / len(links))
    
    with env.begin(write=True, db=split_env) as txn:
        txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))
        txn.put('avg_subgraph_nodes'.encode(), (avg_subgraph_nodes).to_bytes(int.bit_length(avg_subgraph_nodes), byteorder='little'))


def get_average_subgraph_size(sample_size, links, A, rf_list, g, params):
    total_size = 0
    sampled_links = links[np.random.choice(len(links), sample_size)]
    for (n1, n2, r_label) in sampled_links:
        nodes, n_labels = subgraph_extraction_labeling((n1, n2), A, g, params.hop)
        rf = rf_list[[n1, n2]]
        datum = {'nodes': nodes, 'r_label': r_label, 'rf': rf, 'n_labels': n_labels}
        total_size += len(serialize(datum))
    return total_size / sample_size


def intialize_worker(A, rf_list, graph, params):
    global A_, rf_list_, graph_, params_
    A_, rf_list_, graph_, params_ = A, rf_list, graph, params
    

def extract_save_subgraph(args_):
    idx, (n1, n2, r_label) = args_
    nodes, n_labels = subgraph_extraction_labeling((n1, n2), A_, graph_, params_.hop, params_.max_nodes_per_hop)
    rf = rf_list_[[n1, n2]]
    datum = {'nodes': nodes, 'r_label': r_label, 'rf': rf, 'n_labels': n_labels}
    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)


def subgraph_extraction_labeling(ind, A, g, h=1, max_nodes_per_hop=None):
    # source_nei = set(torch.cat(dgl.bfs_nodes_generator(g, ind[0])[: h+1]).numpy())
    # target_nei = set(torch.cat(dgl.bfs_nodes_generator(g, ind[1])[: h+1]).numpy())
    source_nei = get_neighbor_nodes(ind[0], g, h, max_nodes_per_hop)
    target_nei = get_neighbor_nodes(ind[1], g, h, max_nodes_per_hop)
    
    subgraph_nei_nodes = source_nei.intersection(target_nei) - set(ind)
    subgraph_nodes = np.array(list(ind) + list(subgraph_nei_nodes))
    
    subgraph = A[subgraph_nodes, :][:, subgraph_nodes]
    labels, enclosing_subgraph_nodes = node_label(subgraph, max_distance=h)
    
    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes]
    pruned_labels = labels[enclosing_subgraph_nodes]
    return pruned_subgraph_nodes, pruned_labels


def node_label(subgraph, max_distance=1):
    # implementation of the node labeling scheme described in the paper
    dist_to_source = np.clip(ssp.csgraph.dijkstra(remove_nodes(subgraph, [1]), indices=[0], directed=False, unweighted=True, limit=1e2)[:, 1:], 0, 1e2)
    dist_to_target = np.clip(ssp.csgraph.dijkstra(remove_nodes(subgraph, [0]), indices=[0], directed=False, unweighted=True, limit=1e2)[:, 1:], 0, 1e2)
    dist_to_roots = np.array(list(zip(dist_to_source[0], dist_to_target[0])), dtype=int)
    
    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels
    
    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes


def get_neighbor_nodes(root, g, hop, max_nodes_per_hop=None):
    current_roots = [root]
    neighbors = set([root])
    
    for _ in range(0, hop):
        current_neighbors = dgl.bfs_nodes_generator(g, current_roots)
        if len(current_neighbors) >= 2:
            current_neighbors = current_neighbors[1].numpy()
        else:
            break
        
        if max_nodes_per_hop is not None and len(current_neighbors) > max_nodes_per_hop:
            current_neighbors = np.random.choice(current_neighbors, max_nodes_per_hop, replace=False)
        current_roots = list(set(current_neighbors) - neighbors)
        neighbors = neighbors.union(set(current_neighbors))
    
    return neighbors