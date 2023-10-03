import torch as th
import torch.nn as nn
import torch.nn.functional as F
from cogdl.layers import GCNLayer
from cogdl.layers import GATLayer
from aug import aug

# Multi-layer Graph Convolutional Networks
# class GCN(nn.Module):
#     def __init__(self, in_dim, out_dim, act_fn, num_layers=1):
#         super(GCN, self).__init__()

#         self.num_layers = num_layers
#         self.convs = nn.ModuleList()
#         self.convs.append(
#                 GCNLayer(in_features=in_dim, out_features=out_dim, residual=False, norm=None, activation=act_fn)
#             )
#         for _ in range(1, num_layers):
#             self.convs.append(
#                 GCNLayer(in_features=out_dim, out_features=out_dim, residual=False, norm=None, activation=act_fn)
#             )

#     def forward(self, graph, feat):
#         for i in range(self.num_layers):
#             feat = self.convs[i](graph, feat)

#         return feat
class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn, num_layers=1):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(
                GATLayer(in_feats=in_dim, out_feats=out_dim, nhead=1,alpha=0.2, attn_drop=0, activation=None, residual=False)
            )
        for _ in range(1, num_layers):
            self.convs.append(
                GATLayer(in_feats=out_dim, out_feats=out_dim, nhead=1,alpha=0.2, attn_drop=0, activation=None, residual=False)
            )

    def forward(self, graph, feat):
        for i in range(self.num_layers):
            feat = self.convs[i](graph, feat)

        return feat
# Multi-layer(2-layer) Perceptron
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, in_dim)

    def forward(self, x):
        z = F.elu(self.fc1(x))
        return self.fc2(z)

class DotEdgeDecoder(nn.Module):
    """Simple Dot Product Edge Decoder"""

    def __init__(self):
        super().__init__()

    def forward(self, z, edge, sigmoid=True):
        x = F.normalize(z[edge[0]], p=2, dim=-1)
        y = F.normalize(z[edge[1]], p=2, dim=-1)
        ret = (x*y).sum(-1)

        if sigmoid:
            return ret.sigmoid()
        else:
            return ret

class Grace(nn.Module):
    r"""
        GRACE model
    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.
    out_dim: int
        Output feature size.
    num_layers: int
        Number of the GNN encoder layers.
    act_fn: nn.Module
        Activation function.
    temp: float
        Temperature constant.
    """

    def __init__(self, in_dim, hid_dim, out_dim, act_fn, temp, num_layers, drop_edge_rate, drop_node_rate):
        super(Grace, self).__init__()
        self.encoder = GAT(in_dim=in_dim, out_dim=out_dim, act_fn=act_fn, num_layers=num_layers)
        self.edge_decoder = DotEdgeDecoder()
        self.temp = temp
        self.drop_edge_rate = drop_edge_rate
        self.drop_node_rate = drop_node_rate
        self.proj = MLP(hid_dim, out_dim)
        self.negative_sampler = random_negative_sampler

    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = th.mm(z1, z2.t())
        return s

    def get_loss(self, z1, z2):
        # calculate SimCLR loss
        f = lambda x: th.exp(x / self.temp)

        refl_sim = f(self.sim(z1, z1))  # intra-view pairs
        between_sim = f(self.sim(z1, z2))  # inter-view pairs

        # between_sim.diag(): positive pairs
        x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        loss = -th.log(between_sim.diag() / x1)

        return loss

    def forward(self, graph, feat):
        # augument
        graph1, feat1, edge_mask1 = aug(graph, feat, self.drop_node_rate, self.drop_edge_rate)
        graph2, feat2, edge_mask2 = aug(graph, feat, self.drop_node_rate, self.drop_edge_rate)
        # encoding
        h1 = self.encoder(graph1, feat1)
        h2 = self.encoder(graph2, feat2)
        # edges
        edge_index = th.stack(graph.edge_index) 
        masked_edges = edge_index[:, ~edge_mask1]
        neg_edges = self.negative_sampler(
            edge_index,
            num_nodes=graph.num_nodes,
            num_neg_samples=masked_edges.view(2, -1).size(1),
        ).view_as(masked_edges)

        pos_out = self.edge_decoder(h1, masked_edges)
        neg_out = self.edge_decoder(h1, neg_edges)
        
        pos_exp = th.exp(pos_out/self.temp)
        neg_exp = th.sum(th.exp(neg_out/self.temp))
        l3 = -th.log(pos_exp / (pos_exp + neg_exp) + 1e-15).mean()

        # projection
        z1 = self.proj(h1)
        z2 = self.proj(h2)

        # get loss
        l1 = self.get_loss(z1, z2)
        l2 = self.get_loss(z2, z1)

        ret = (l1 + l2) * 0.5

        return ret.mean()+l3



def random_negative_sampler(edge_index, num_nodes, num_neg_samples):
    neg_edges = th.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index)
    return neg_edges
