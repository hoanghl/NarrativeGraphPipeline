from itertools import combinations
from random import sample

import numpy as np
import torch
import torch.nn as torch_nn
import torch_geometric.nn as torch_g_nn


class GraphBasedLayer(torch_nn.Module):
    def __init__(
        self,
        batch_size,
        l_q,
        l_a,
        d_hid,
        d_bert,
        d_graph,
        n_nodes,
        n_edges,
    ):
        super().__init__()

        self.bz = batch_size
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.d_hid = n_nodes
        self.d_bert = d_bert

        self.lin1 = torch_nn.Linear(d_bert * 2, d_hid, bias=False)

        self.graph = GraphLayer(d_hid, d_graph, n_nodes)

        self.lin2 = torch_nn.Linear(d_bert, l_a, bias=False)
        self.lin3 = torch_nn.Linear(l_q, n_nodes, bias=False)
        self.lin4 = torch_nn.Linear(d_hid, d_bert, bias=False)

        self.edge_indx = self.gen_edges()

    def forward(self, q, c):
        # q : [b, l_q, d_bert]
        # c: [b, n_c, d_bert]

        (
            b,
            n_c,
            _,
        ) = c.size()

        ######################################
        # Use TransformerEncoder to encode q and c
        ######################################
        q_ = torch.mean(q, dim=1).unsqueeze(1).repeat(1, n_c, 1)
        X = torch.cat((c, q_), dim=2)
        # [b, n_c, d_bert*2]
        X = self.lin1(X)
        # [b, n_c, d_hid]

        edge_indx = self.edge_indx[:b, :, :]
        edge_len = torch.IntTensor([self.n_edges]).repeat(b)
        # edge_indx         : [b, 2, n_edges]
        # edge_len          : [b]

        # Create node feat from tensor X
        node_feats = []
        for pair in combinations(range(n_c), 2):
            idx1, idx2 = pair
            node_feats.append(torch.cat([X[:, idx1, :], X[:, idx2, :]], dim=-1).unsqueeze(1))

        node_feats = torch.cat(node_feats, dim=1)
        # [b, n_nodes, d_hid*2]
        node_len = torch.IntTensor([self.n_nodes]).repeat(b)
        # [b]

        ######################################
        # Pass through Graph
        ######################################
        Y = self.graph(node_feats, edge_indx, node_len, edge_len)
        # [b, n_nodes, d_hid]

        Y = self.lin4(Y)
        # [b, n_nodes, d_bert]

        return Y

    def gen_edges(self):
        edge_pair = list(combinations(range(self.n_nodes), 2))
        edges = sample(edge_pair, self.n_edges // 2)

        vertex_s, vertex_d = [], []
        for edge in edges:
            s, d = edge
            vertex_s.append(int(s))
            vertex_d.append(int(d))

            vertex_s.append(int(d))
            vertex_d.append(int(s))

        edge_index = np.array([vertex_s, vertex_d])
        # [2, *]

        edge_index = torch.from_numpy(edge_index).unsqueeze(0).repeat(self.bz, 1, 1)

        return edge_index


class GraphLayer(torch_nn.Module):
    def __init__(self, d_hid, d_graph, n_nodes):

        super().__init__()

        self.linear = torch_nn.Linear(d_hid * 2, d_graph)

        # GCN
        self.gcn1 = torch_g_nn.GCNConv(d_graph, d_graph)
        self.gcn2 = torch_g_nn.GCNConv(d_graph, d_graph // 2)
        self.gcn3 = torch_g_nn.GCNConv(d_graph // 2, d_graph // 2)
        self.gcn4 = torch_g_nn.GCNConv(d_graph // 2, d_graph // 4)
        self.act_leakRelu = torch_nn.LeakyReLU(inplace=True)
        self.batchnorm = torch_nn.BatchNorm1d(n_nodes)

        self.linear2 = torch_nn.Linear(d_graph // 4, d_hid, bias=False)

    def forward(self, node_feat, edge_indx, node_len, edge_len):
        # node_feat : [b, n_nodes, d_hid * 2]
        # edge_indx : [b, 2, n_edges]
        # node_len  : [b]
        # edge_len  : [b]

        b, n_nodes, d = node_feat.shape
        d_hid = d // 2

        def gcn(node_feats, edge_indx):
            X = self.gcn1(node_feats, edge_indx)
            X = self.act_leakRelu(X)

            X = self.gcn2(X, edge_indx)
            X = self.act_leakRelu(X)

            X = self.gcn3(X, edge_indx)
            X = self.act_leakRelu(X)

            X = self.gcn4(X, edge_indx)
            X = self.act_leakRelu(X)

            return X

        node_feat = self.linear(node_feat)

        node_feat, edge_indx, _ = self.batchify(node_feat, node_len, edge_indx, edge_len)

        X = gcn(node_feat, edge_indx)
        # [n_nodes * b, d_graph//4]

        X = self.linear2(X).view(b, n_nodes, d_hid)
        # [b, n_nodes, d_hid]
        X = self.batchnorm(X)

        return X

    def batchify(
        self,
        node_feat,
        node_len,
        edge_indx,
        edge_len,
    ) -> tuple:
        """Convert batch of node features and edge indices into a big graph"""
        # node_feat : [b, n_nodes, d_hid*2]
        # node_len  : [b]
        # edge_indx : [b, 2, *]
        # edge_len  : [b]
        batch = node_feat.shape[0]

        accum = 0
        final_edge_indx = None
        final_node_feat = None
        batch_indx = []
        for b in range(batch):
            ## 1. accummulate node feature
            ## 1.1. get node feat of that batch and remove padding
            node_feat_ = node_feat[b, : node_len[b].item(), :].squeeze(0)

            ## 1.3. Concate into 'final_node_feat'
            if final_node_feat is None:
                final_node_feat = node_feat_
            else:
                final_node_feat = torch.vstack((final_node_feat, node_feat_))

            ## 2. accummulate edge indx
            ## 2.1. get edge indx of that batch and remove padding
            edge_indx_ = edge_indx[b, :, : edge_len[b].item()].squeeze(0)

            ## 2.2. Increment index of that edge indx by accum
            increment = torch.tensor([accum], device=edge_indx_.device).repeat(edge_indx_.shape)
            edge_indx_ = edge_indx_ + increment

            ## 2.3. Concate into 'final_edge_indx'
            if final_edge_indx is None:
                final_edge_indx = edge_indx_
            else:
                final_edge_indx = torch.hstack((final_edge_indx, edge_indx_))

            ## 3. Update batch_indx and accum
            batch_indx = batch_indx + [b] * (node_len[b].item())
            accum += node_len[b].item()

        return final_node_feat, final_edge_indx.long(), torch.LongTensor(batch_indx)
