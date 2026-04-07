import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if type(output) is tuple:
        output = output[0]

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_multiple_labels(output, targets, topk=(1,)):
    assert output.shape[1] >= targets.shape[1], f"Predictions shape {output.shape} must match targets shape {targets.shape}"

    output = output[:, :targets.shape[1]]

    maxk = max(topk)
    _, preds = output.topk(maxk, 1, True, True)

    batch_size = len(output)
    with torch.no_grad():
        res = [[] for _ in range(len(topk))]

        for pred, target in zip(preds, targets):
            target_nz = torch.nonzero(target)
            for i, k in enumerate(topk):
                correct = any(p in target_nz for p in  pred[:k])
                res[i].append(correct)
    return [sum(r) / len(r) * 100. for r in res]

def dist_text_embed_one_view(motion_feats, text_feats):
    motion_feats = nn.functional.normalize(motion_feats, dim=1)
    text_feats = nn.functional.normalize(text_feats, dim=1)

    if True:
        sim_dists = torch.mm(motion_feats, text_feats.t())


    else:
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

        L = len(text_feats)
        euc_dists = torch.zeros(1, L).to(motion_feats.device)
        for idx in range(L):
            text_feats_it = text_feats[idx:idx+1, :]
            # text_feats_it = nn.functional.normalize(text_feats_it, dim=1)
            # euclidean_distance = nn.functional.pairwise_distance(motion_feats, text_feats_it)
            # euclidean_distance = torch.mm(motion_feats, text_feats_it.t())
            euclidean_distance = cos(motion_feats, text_feats_it)
            euc_dists[0, idx] = euclidean_distance[0]

    return sim_dists

def sim_dists_text_embed(output):
    if type(output) is not tuple:
        print("wrong output format : ", type(output))
    
    # All motions compared with 1 view
    if len(output[1].shape) == 4:
        motion_feats_all = output[1]
        text_feats = output[2]
        L = len(text_feats)
        V = motion_feats_all.shape[1]
        sim_dists = torch.zeros(V, L).to(motion_feats_all.device)
        for iv in range(V):
            sim_dists[iv] = dist_text_embed_one_view(motion_feats_all[:, iv], text_feats)
        sim_dists = sim_dists.mean(dim=0)
        sim_dists = sim_dists.unsqueeze(dim=0)

    # All motions and texts compared by views
    elif type(output[2]) is list and output[1].shape[1] == len(output[2]):
        motion_feats_all = output[1]
        text_feats = output[2]
        L = len(text_feats[0]) # collect length of first text_features
        V = motion_feats_all.shape[1]
        sim_dists = torch.zeros(V, L).to(motion_feats_all.device)
        for iv in range(V):
            sim_dists[iv] = dist_text_embed_one_view(motion_feats_all[:, iv], text_feats[iv])
        sim_dists = sim_dists.mean(dim=0)
        sim_dists = sim_dists.unsqueeze(dim=0)

    # 1 motion compared with all texts
    elif len(output[2].shape) == 3:
        motion_feats = output[0]
        text_feats = output[2]
        V, L, _ = text_feats.shape # collect lengths of first text_features
        sim_dists = torch.zeros(V, L).to(motion_feats.device)
        for iv in range(V):
            sim_dists[iv] = dist_text_embed_one_view(motion_feats, text_feats[iv])
        sim_dists = sim_dists.mean(dim=0)
        sim_dists = sim_dists.unsqueeze(dim=0)

    # single motion and text
    else:
        motion_feats = output[0]
        text_feats = output[2]
        sim_dists = dist_text_embed_one_view(motion_feats, text_feats)

    return sim_dists

def accuracy_text_embed(output, target, topk=(1,)):
    sim_dists = sim_dists_text_embed(output)
    return accuracy(sim_dists, target, topk), sim_dists

def map_gtidx_to_color(feats_gt_idxs, label_map):
    # get unique ids
    onlyidxs = sorted(set(feats_gt_idxs))

    # color idxs with labels
    coloridxs = list(range(len(onlyidxs)))
    colorlabels = [label_map[idx] for idx in onlyidxs]

    # labels as color index
    gt_to_color = dict(zip(onlyidxs, coloridxs))
    all_color_idxs = [gt_to_color[gt_idx] for gt_idx in feats_gt_idxs]
    return all_color_idxs, colorlabels

def top_k_by_action(scores, labels, k=1):
    labels = np.array(labels)[:, np.newaxis]
    max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]

    match_by_action = {}

    for i in range(len(labels)):
        label = labels[i][0]
        pred = max_k_preds[i]

        if label not in match_by_action.keys():
            match_by_action[label] = []
        match_by_action[label].append(np.logical_or.reduce(pred == np.array(label)))

    topk_by_action = {}
    for action in match_by_action.keys():
        topk_by_action[action] = sum(match_by_action[action]) / len(match_by_action[action])

    return topk_by_action

def cosine_loss(pred, target):
    """
    pred and target are both [batch_size, 2], normalized to unit vectors.
    """
    return 1 - F.cosine_similarity(pred, target, dim=1).mean()

def mse_vector_loss(pred, target):
    return F.mse_loss(pred, target)

