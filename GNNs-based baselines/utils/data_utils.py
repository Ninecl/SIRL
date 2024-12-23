import os
import torch

def read_triplets2id(path, mode, entity2id=dict(), relation2id = dict(), with_head=False, allow_emerging=False):
    """读取文本三元组，返回转化为id后的三元组和id映射字典

    Args:
        path (str): 读入路径
        mode (str): 可选为hrt与htr，表示输入三元组格式
        entity2id (dict, optional): 输入实体id映射，若为第一次读取则默认为空字典. Defaults to dict().
        relation2id (dict, optional): 输入关系id映射，若为第一次读取则默认为空字典. Defaults to dict().
        with_head (bool, optional): 文本是否带头. Defaults to False.

    Returns:
        triplets (list): 转换为id后的三元组，默认为 (h, r, t)格式
        entity2id (dict): 实体映射字典
        relation2id (dict): 关系映射字典
    """
    # 三元组转化为id后的list, (h, r, t)形式
    triplets = []
    
    with open(path, 'r') as f:
        data = f.readlines() if not with_head else f.readlines()[1: ]
        lines = [line.strip().split() for line in data]
        
        ent_cnt = len(entity2id)
        rel_cnt = len(relation2id)
        for line in lines:
            if mode == 'hrt':
                h, r, t = line
            elif mode == 'htr':
                h, t, r = line
            else:
                raise "ERROR: illegal triplet form"
            
            if not allow_emerging:
                assert (h in entity2id) and (r in relation2id) and (t in entity2id)
            else:
                if h not in entity2id:
                    entity2id[h] = ent_cnt
                    ent_cnt += 1
                if t not in entity2id:
                    entity2id[t] = ent_cnt
                    ent_cnt += 1
                if r not in relation2id:
                    relation2id[r] = rel_cnt
                    rel_cnt += 1
            
            triplets.append([entity2id[h], relation2id[r], entity2id[t]])
    
    return triplets, entity2id, relation2id


def collate_function(samples):
    return torch.LongTensor(samples)


def sample_neg_triplets(triplets, num_ent, num_neg, device=None):
    neg_triplets = torch.LongTensor(triplets).unsqueeze(dim=1).repeat(1, num_neg, 1)
    rand_result = torch.rand((len(triplets), num_neg))
    perturb_head = rand_result < 0.5
    perturb_tail = rand_result >= 0.5
    rand_idxs = torch.randint(low=0, high = num_ent-1, size = (len(triplets), num_neg))
    neg_triplets[:,:,0][perturb_head] = rand_idxs[perturb_head]
    neg_triplets[:,:,2][perturb_tail] = rand_idxs[perturb_tail]
    neg_triplets = neg_triplets.view(-1, 3)
    pos_triplets = torch.LongTensor(triplets)
    
    if device is not None:
        pos_triplets = pos_triplets.to(device)
        neg_triplets = neg_triplets.to(device)
    return pos_triplets, neg_triplets


def write_tensor_scores(path, scores):
    with open(path, 'w') as f:
        for score in scores:
            f.write(f'{score.item()}\n')
        f.close()