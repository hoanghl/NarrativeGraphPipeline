from itertools import combinations

import torch
import torch.nn as torch_nn

from models.layers.reasoning_layer.sub_layers import GraphLayer, Memory


class GraphBasedMemoryLayer(torch_nn.Module):
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

        self.n_nodes = n_nodes
        self.d_hid = n_nodes
        self.d_bert = d_bert

        self.lin1 = torch_nn.Linear(d_bert * 2, d_hid, bias=False)

        self.graph = GraphLayer(d_hid, d_graph, n_nodes)
        self.memory = Memory(batch_size, n_nodes, d_hid, n_edges)

        self.lin2 = torch_nn.Linear(d_bert, l_a, bias=False)
        self.lin3 = torch_nn.Linear(l_q, n_nodes, bias=False)
        self.lin4 = torch_nn.Linear(d_hid, d_bert, bias=False)

    def forward(self, q, c):
        # q : [b, l_q, d_bert]
        # c: [b, n_c, d_bert]

        (
            b,
            n_c,
            _,
        ) = c.size()

        ######################################
        # Use TransformerEncoder to encode
        # question and c
        ######################################
        q_ = torch.mean(q, dim=1).unsqueeze(1).repeat(1, n_c, 1)
        X = torch.cat((c, q_), dim=2)
        # [b, n_c, d_bert*2]
        X = self.lin1(X)
        # [b, n_c, d_hid]

        ######################################
        # Get node feat and edge indx from memory
        # and combine with X to create tensors for Graph
        ######################################
        # Get things from memory
        node_feats_mem, edge_indx, edge_len = self.memory.gets()
        # node_feats_mem    : [batch, n_nodes, d_hid]
        # edge_indx         : [batch, 2, n_edges]
        # edge_len          : [batch]

        node_feats_mem = node_feats_mem[:b, :, :]
        edge_indx = edge_indx[:b, :, :]
        edge_len = edge_len[:b]
        # node_feats_mem    : [b, n_nodes, d_hid]
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

        # Concat 'node_feats' with 'node_feats_mem'
        node_feats = torch.cat((node_feats, node_feats_mem), dim=2)
        # [b, n_nodes, d_hid*3]

        ######################################
        # Pass through Graph
        ######################################
        Y = self.graph(node_feats, edge_indx, node_len, edge_len)
        # [b, n_nodes, d_hid]

        ######################################
        # Update memory
        ######################################
        self.memory.update_mem(Y)

        Y = self.lin4(Y)
        # [b, n_nodes, d_bert]

        return Y

    def save_memory(self):
        self.memory.save_memory()
