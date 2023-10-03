# Data augmentation on graphs via edge dropping and feature masking

import cogdl
import numpy as np
import torch as th


def aug(graph, x, feat_drop_rate, edge_mask_rate):
    n_node = graph.num_nodes

    edge_mask = mask_edge(graph, edge_mask_rate)
    feat = drop_feature(x, feat_drop_rate)

    src = graph.edge_index[0]
    dst = graph.edge_index[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]
    edges = th.cat((nsrc.unsqueeze(0), ndst.unsqueeze(0)), dim=0)
    ng = cogdl.data.Graph(edge_index=edges, num_nodes=n_node)
    ng = ng.to(graph.device)
    ng = ng.padding_self_loops()

    return ng, feat, edge_mask


def drop_feature(x, drop_prob):
    drop_mask = (
        th.empty((x.size(1),), dtype=th.float32, device=x.device).uniform_(0, 1)
        < drop_prob
    )
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def mask_edge(graph, mask_prob):
    E = graph.num_edges

    mask_rates = th.FloatTensor(np.ones(E) * mask_prob)
    masks = th.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx
