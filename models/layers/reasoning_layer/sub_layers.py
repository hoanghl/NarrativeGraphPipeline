from typing import Any
from itertools import combinations
from random import sample

from torch.nn.parameter import Parameter
import torch_geometric.nn as torch_g_nn
import torch.nn as torch_nn
import torch
import numpy as np


class GraphLayer(torch_nn.Module):
    def __init__(self, d_hid, d_graph, n_nodes):

        super().__init__()

        self.linear = torch_nn.Linear(d_hid * 3, d_graph)

        # GCN
        self.gcn1 = torch_g_nn.GCNConv(d_graph, d_graph)
        self.gcn2 = torch_g_nn.GCNConv(d_graph, d_graph // 2)
        self.gcn3 = torch_g_nn.GCNConv(d_graph // 2, d_graph // 2)
        self.gcn4 = torch_g_nn.GCNConv(d_graph // 2, d_graph // 4)
        self.act_leakRelu = torch_nn.LeakyReLU(inplace=True)
        self.batchnorm = torch_nn.BatchNorm1d(n_nodes)

        self.linear2 = torch_nn.Linear(d_graph // 4, d_hid, bias=False)

    def forward(self, node_feat, edge_indx, node_len, edge_len):
        # node_feat : [b, n_nodes, d_hid*3]
        # edge_indx : [b, 2, n_edges]
        # node_len  : [b]
        # edge_len  : [b]

        b, n_nodes, d_hid = node_feat.shape
        d_hid = d_hid // 3

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

        node_feat, edge_indx, _ = self.batchify(
            node_feat, node_len, edge_indx, edge_len
        )

        X = gcn(node_feat, edge_indx)
        # [n_nodes * b, d_graph//4]

        X = self.linear2(X).view(b, n_nodes, d_hid)
        # [b, n_nodes, d_hid]
        X = self.batchnorm(X)

        return X

    def batchify(
        self,
        node_feat: torch.Tensor,
        node_len: torch.Tensor,
        edge_indx: torch.Tensor,
        edge_len: torch.Tensor,
    ) -> tuple:
        """Convert batch of node features and edge indices into a big graph"""
        # node_feat : [b, n_nodes, d_hid*3]
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
            increment = torch.Tensor([accum]).repeat(edge_indx_.shape)
            edge_indx_ = edge_indx_ + increment.to(edge_indx_.get_device())

            ## 2.3. Concate into 'final_edge_indx'
            if final_edge_indx is None:
                final_edge_indx = edge_indx_
            else:
                final_edge_indx = torch.hstack((final_edge_indx, edge_indx_))

            ## 3. Update batch_indx and accum
            batch_indx = batch_indx + [b] * (node_len[b].item())
            accum += node_len[b].item()

        return final_node_feat, final_edge_indx.long(), torch.LongTensor(batch_indx)


class Memory(torch_nn.Module):
    def __init__(
        self,
        batch_size,
        n_nodes,
        d_hid,
        n_edges,
    ):

        super().__init__()

        self.batch = batch_size
        self.n_nodes = n_nodes
        self.d_hid = d_hid
        self.n_edges = n_edges

        self.edge_len = torch.IntTensor([n_edges]).repeat(self.batch)

        ## If load_statedict occurs, it will automatically load the following attributes
        self.node_feats_mem = Parameter(
            torch.rand(self.batch, self.n_nodes, self.d_hid), requires_grad=False
        )
        self.edge_index = Parameter(self.gen_edges(), requires_grad=False)

    def forward(self):
        pass

    def update_mem(self, Y):
        """Update memory with given tensor

        Args:
            Y (Tensor): output of Graph module
        """
        # Y: [b, n_nodes, d_hid]
        b = Y.shape[0]
        if b < self.batch:
            tmp = self.node_feats_mem[b:, :, :]
            Y = torch.cat((Y, tmp), dim=0)
        # Y: [batch, n_nodes, d_hid]

        self.node_feats_mem = torch.nn.parameter.Parameter(
            Y.detach(), requires_grad=False
        )

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

        edge_index = torch.from_numpy(edge_index).unsqueeze(0).repeat(self.batch, 1, 1)

        return edge_index

    def gets(self):
        return self.node_feats_mem, self.edge_index, self.edge_len
