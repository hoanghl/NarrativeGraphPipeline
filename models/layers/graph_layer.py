import torch
import torch.nn as torch_nn
import torch_geometric.nn as torch_g_nn


class GraphBasedReasoningLayer(torch_nn.Module):
    def __init__(self, d_hid, d_bert, d_graph, n_propagations, n_edges, dropout):
        super().__init__()

        self.n_edges = n_edges
        self.d_hid = d_hid

        self.graph = GraphLayer(d_hid, d_graph, n_propagations)
        self.lin1 = torch_nn.Linear(d_bert, d_hid, bias=False)
        self.lin2 = torch_nn.Linear(d_hid, d_bert, bias=False)

    def forward(self, c_hid, edge_indx):
        # c_hid: [b, nc, lc, d_bert]
        # edge_indx: [b, 2, n_edges]

        bz, nc, _, _ = c_hid.size()

        node_len = torch.full((bz,), nc, dtype=torch.long, device=c_hid.device)
        node_feats = c_hid[:, :, 0]  # Extract CLS embedding only
        node_feats = self.lin1(node_feats)
        # [b, nc, d_hid]

        edge_len = torch.full((bz,), self.n_edges, dtype=torch.long, device=c_hid.device)

        ######################################
        # Pass through Graph
        ######################################
        outputs = self.graph(node_feats, edge_indx, node_len, edge_len)
        # [b, nc, d_hid]

        outputs = self.lin2(outputs)
        # [b, nc, d_bert]

        return outputs


class GraphLayer(torch_nn.Module):
    def __init__(self, d_hid, d_graph, n_propagations):

        super().__init__()

        self.n_propagations = n_propagations

        self.lin1 = torch_nn.Linear(d_hid, d_graph)

        # GCN
        self.gcn1 = torch_g_nn.GCNConv(d_graph, d_graph)
        self.gcn2 = torch_g_nn.GCNConv(d_graph, d_graph)

        self.act_leakRelu = torch_nn.LeakyReLU(inplace=True)

        self.lin2 = torch_nn.Linear(d_graph, d_hid, bias=False)

    def forward(self, node_feat, edge_indx, node_len, edge_len):
        # node_feat : [b, nc, d_hid]
        # edge_indx : [b, 2, n_edges]
        # node_len  : [b]
        # edge_len  : [b]

        b, nc, d_hid = node_feat.size()

        def gcn(node_feats, edge_indx):
            X = self.gcn1(node_feats, edge_indx)
            X = self.act_leakRelu(X)

            X = self.gcn2(X, edge_indx)
            X = self.act_leakRelu(X)

            return X

        node_feat = self.lin1(node_feat)

        node_feat, edge_indx, _ = self.batchify(node_feat, node_len, edge_indx, edge_len)

        X = node_feat
        for _ in range(self.n_propagations):
            X = gcn(X, edge_indx)
            # [nc * b, d_graph]

        X = self.lin2(X).view(b, nc, d_hid)
        # [b, nc, d_hid]

        return X

    def batchify(
        self,
        node_feat,
        node_len,
        edge_indx,
        edge_len,
    ) -> tuple:
        """Convert batch of node features and edge indices into a big graph"""
        # node_feat : [b, nc, d_hid*2]
        # node_len  : [b]
        # edge_indx : [b, 2, *]
        # edge_len  : [b]
        batch = node_feat.shape[0]

        accum = 0
        final_node_feat = []
        final_edge_indx = []
        batch_indx = []
        for b in range(batch):
            ## 1. accummulate node feature
            ## 1.1. get node feat of that batch and remove padding
            node_feat_ = node_feat[b, : node_len[b], :].squeeze(0)

            ## 1.3. Concate into 'final_node_feat'
            final_node_feat.append(node_feat_)

            ## 2. accummulate edge indx
            ## 2.1. get edge indx of that batch and remove padding
            edge_indx_ = edge_indx[b, :, : edge_len[b]].squeeze(0)

            ## 2.2. Increment index of that edge indx by accum
            increment = torch.full(edge_indx_.size(), accum, device=edge_indx_.device)
            edge_indx_ = edge_indx_ + increment

            ## 2.3. Concate into 'final_edge_indx'
            final_edge_indx.append(edge_indx_)

            ## 3. Update batch_indx and accum
            batch_indx = batch_indx + [b] * (node_len[b].item())
            accum += node_len[b]

        return (
            torch.cat(final_node_feat, dim=0),
            torch.cat(final_edge_indx, dim=-1).long(),
            torch.tensor(batch_indx, dtype=torch.long),
        )
