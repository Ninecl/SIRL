import os
import dgl
import json
import torch
import pickle

import numpy as np

from scipy.sparse import csc_matrix


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def dump_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)
        

def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)


def deserialize(data):
    data_tuple = pickle.loads(data)
    return data_tuple


def read_triplets(path, mode, with_head=False):
    """read triplets from file

    Args:
        path (str): file path
        mode (str): the triplets form, hrt or htr
        with_head (bool, optional): whether contrain a head about the number of triplets. Defaults to False.

    Returns:
        triplets (list): list of triplets in the form of hrt
    """
    triplets = []
    with open(path, 'r') as f:
        data = f.readlines() if not with_head else f.readlines()[1: ]
        lines = [line.strip().split() for line in data]
        
        for line in lines:
            if mode == 'hrt':
                h, r, t = line
            elif mode == 'htr':
                h, t, r = line
            else:
                raise "ERROR: illegal triplet form"
            triplets.append([h, r, t])
    return triplets


def read_se_support_triplets(path_dics, batch_idx, mode, with_head=False):
    """read triplets from file

    Args:
        path (str): file path
        mode (str): the triplets form, hrt or htr
        with_head (bool, optional): whether contrain a head about the number of triplets. Defaults to False.

    Returns:
        triplets (list): list of triplets in the form of hrt
    """
    triplets = []
    for i in range(0, batch_idx + 1):
        path = path_dics[i]['support']
        with open(path, 'r') as f:
            data = f.readlines() if not with_head else f.readlines()[1: ]
            lines = [line.strip().split() for line in data]
            
            for line in lines:
                if mode == 'hrt':
                    h, r, t = line
                elif mode == 'htr':
                    h, t, r = line
                else:
                    raise "ERROR: illegal triplet form"
                triplets.append([h, r, t])
    return triplets


def process_triplets(support_triplets, query_triplets, saved_relation2id=None):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id
    triplets2id_data = {}
    triplets_data = {'support': support_triplets, 'query': query_triplets}

    ent = 0
    rel = 0

    for triplets_type, triplets in triplets_data.items():
        data = []
        for h, r, t in triplets:
            # Make sure the r is seen
            if saved_relation2id is not None:
                assert r in relation2id
                
            if h not in entity2id:
                entity2id[h] = ent
                ent += 1
            if t not in entity2id:
                entity2id[t] = ent
                ent += 1
            if r not in relation2id:
                relation2id[r] = rel
                rel += 1
            data.append([entity2id[h], entity2id[t], relation2id[r]])
        triplets2id_data[triplets_type] = np.array(data)

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets2id_data['support'][:, 2] == i)
        rows = triplets2id_data['support'][:, 0][idx].squeeze(1)
        cols = triplets2id_data['support'][:, 1][idx].squeeze(1)
        data = np.ones(len(idx), dtype=np.uint8)
        adj_list.append(csc_matrix((data, (rows, cols)), shape=(len(entity2id), len(entity2id))))
    
    # Construct list of relation feature for each entity, also is constrcuted only from the train data.
    rf_list = np.zeros((len(entity2id), len(relation2id)))
    for triplets2id in triplets2id_data['support']:
        h, t, r = triplets2id
        rf_list[h][r] += 1
        rf_list[t][r] += 1

    return adj_list, rf_list, triplets2id_data, entity2id, relation2id
