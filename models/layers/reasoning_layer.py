import torch
import torch.nn as torch_nn
from torch.nn.parameter import Parameter


class SelfAttnQueryRetrv(torch_nn.Module):
    def __init__(self, l_q, l_c, n_heads, d_hid, device):
        super().__init__()

        self.l_c = l_c
        self.l_q = l_q

        d_k = d_hid // n_heads
        self.W_q = Parameter(torch.rand(n_heads, d_hid, d_k, device=device))
        self.W_c = Parameter(torch.rand(n_heads, d_hid, d_k, device=device))
        self.W_m = Parameter(torch.rand(n_heads, l_c, l_c, device=device))
        self.W_0 = Parameter(torch.rand(n_heads, 1, device=device))
        self.ff1 = torch_nn.Sequential(
            torch_nn.Linear(self.l_c, self.l_c),
            torch_nn.Tanh(),
            torch_nn.BatchNorm1d(l_q),
        )

    def forward(self, q, c, m):
        # q: [b, l_q, d_hid]
        # c: [b, n_c, l_c, d_hid]
        # m: [l_c, l_c]

        b, n_c, _, _ = c.size()

        q_proj = q.unsqueeze(1) @ self.W_q
        # [b, n_heads, l_q, d_k]
        c_proj = c.unsqueeze(2) @ self.W_c
        # [b, n_c, n_heads, l_c, d_k]
        m_proj = m.view(1, 1, self.l_c, self.l_c) @ self.W_m
        # [b, n_heads, l_c, l_c]

        product = q_proj.unsqueeze(1) @ c_proj.transpose(-1, -2)
        # [b, n_c, n_heads, l_q, l_c]
        product = self.ff1(product.view(-1, self.l_q, self.l_c)).view(
            b, n_c, -1, self.l_q, self.l_c
        )
        # [b, n_c, n_heads, l_q, l_c]
        weights = torch.softmax(torch.sum(product, dim=-2), dim=-1)
        # [b, n_c, n_heads, l_c]
        w_sum = weights.unsqueeze(3) @ m_proj.unsqueeze(1)
        # [b, n_c, n_heads, 1, l_c]
        output = w_sum.squeeze(-2).transpose(-1, -2) @ self.W_0
        # [b, n_c, l_c, 1]

        return torch.softmax(output.squeeze(-1), dim=-1)
        # [b, n_c, l_c]


class SelfAttnContxRetrv(torch_nn.Module):
    def __init__(self, l_q, l_c, n_heads, d_hid, device):
        super().__init__()

        self.l_q = l_q
        self.l_c = l_c

        d_k = d_hid // n_heads
        self.W_q = Parameter(torch.rand(n_heads, d_hid, d_k, device=device))
        self.W_c = Parameter(torch.rand(n_heads, d_hid, d_k, device=device))
        self.W_m = Parameter(torch.rand(n_heads, l_q, l_q, device=device))
        self.W_0 = Parameter(torch.rand(n_heads, 1, device=device))
        self.ff1 = torch_nn.Sequential(
            torch_nn.Linear(self.l_q, self.l_q),
            torch_nn.Tanh(),
            torch_nn.BatchNorm1d(l_c),
        )

    def forward(self, q, c, m):
        # q: [b, l_q, d_hid]
        # c: [b, n_c, l_c, d_hid]
        # m: [l_q, l_q]

        b, n_c, _, _ = c.size()

        q_proj = q.unsqueeze(1) @ self.W_q
        # [b, n_heads, l_q, d_k]
        c_proj = c.unsqueeze(2) @ self.W_c
        # [b, n_c, n_heads, l_c, d_k]
        m_proj = m.view(1, 1, self.l_q, self.l_q) @ self.W_m
        # [b, n_heads, l_q, l_q]

        product = c_proj @ q_proj.unsqueeze(1).transpose(-1, -2)
        # [b, n_c, n_heads, l_c, l_q]
        product = self.ff1(product.view(-1, self.l_c, self.l_q)).view(
            b, n_c, -1, self.l_c, self.l_q
        )
        # [b, n_c, n_heads, l_c, l_q]
        weights = torch.softmax(torch.sum(product, dim=-2), dim=-1)
        # [b, n_c, n_heads, l_q]
        w_sum = weights.unsqueeze(3) @ m_proj.unsqueeze(1)
        # [b, n_c, n_heads, 1, l_q]
        output = w_sum.squeeze(-2).transpose(-1, -2) @ self.W_0
        # [b, n_c, l_q, 1]

        return torch.softmax(output.squeeze(-1).sum(1), dim=-1)
        # [b, l_q]


class Reasoning(torch_nn.Module):
    def __init__(self, l_q, l_c, n_heads, d_hid, dropout, device):
        super().__init__()

        self.query_retrv = SelfAttnQueryRetrv(
            l_q=l_q, l_c=l_c, n_heads=n_heads, d_hid=d_hid, device=device
        )
        self.mem_c_retrv = Parameter(torch.rand(l_c, l_c))
        self.contx_retrv = SelfAttnContxRetrv(
            l_q=l_q, l_c=l_c, n_heads=n_heads, d_hid=d_hid, device=device
        )
        self.mem_q_retrv = Parameter(torch.rand(l_q, l_q))

        self.ff1 = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid * 2),
            torch_nn.Tanh(),
            torch_nn.Dropout(dropout),
        )
        self.lin1 = torch_nn.Linear(2 * d_hid, 2 * d_hid)

    def forward(self, q, c):
        # q: [b, l_q, d_hid]
        # c: [b, n_c, l_c, d_hid]

        b, n_c, _, d_hid = c.size()

        ## Retrieve filters from long-term memory and apply them
        c_filter = self.query_retrv(q=q, c=c, m=self.mem_c_retrv)
        # [b, n_c, l_c]
        q_filter = self.contx_retrv(q=q, c=c, m=self.mem_q_retrv)
        # [b, l_q]
        c = c * c_filter.unsqueeze(-1)
        # [b, n_c, l_c, d_hid]
        q = q * q_filter.unsqueeze(-1)
        # [b, l_q, d_hid]

        ## Use CoAttention to sequentially update context
        c_ = []
        for i in range(n_c):
            c_i = c[:, i]

            A = self.ff1(q) @ self.lin1(self.ff1(c_i)).transpose(-1, -2)
            # [b, l_q, l_c]

            q = torch.softmax(A, dim=-1) @ c_i
            # [b, l_q, d_hid]
            c_i = torch.softmax(A, dim=-2).transpose(-1, -2) @ q
            # [b, l_c, d_hid]

            c_.append(c_i.unsqueeze(1))

        Y = torch.cat(c_, dim=1).view(b, -1, d_hid)
        # [b, n_c*l_c, d_hid]

        return Y
