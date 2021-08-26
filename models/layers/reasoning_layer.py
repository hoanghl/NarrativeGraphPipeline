import torch
import torch.nn as torch_nn
from torch.nn.parameter import Parameter


class SelfAttnQueryRetrv(torch_nn.Module):
    def __init__(self, lq, lc, n_heads, d_hid, device):
        super().__init__()

        self.lc = lc
        self.lq = lq

        d_k = d_hid // n_heads
        self.W_q = Parameter(torch.rand(n_heads, d_hid, d_k, device=device))
        self.W_c = Parameter(torch.rand(n_heads, d_hid, d_k, device=device))
        self.W_m = Parameter(torch.rand(n_heads, lc, lc, device=device))
        self.W_0 = Parameter(torch.rand(n_heads, 1, device=device))
        self.ff1 = torch_nn.Sequential(
            torch_nn.Linear(self.lc, self.lc),
            torch_nn.Tanh(),
            torch_nn.BatchNorm1d(lq),
        )

    def forward(self, q, c, m):
        # q: [b, lq, d_hid]
        # c: [b, n_c, lc, d_hid]
        # m: [lc, lc]

        b, n_c, _, _ = c.size()

        q_proj = q.unsqueeze(1) @ self.W_q
        # [b, n_heads, lq, d_k]
        c_proj = c.unsqueeze(2) @ self.W_c
        # [b, n_c, n_heads, lc, d_k]
        m_proj = m.view(1, 1, self.lc, self.lc) @ self.W_m
        # [b, n_heads, lc, lc]

        product = q_proj.unsqueeze(1) @ c_proj.transpose(-1, -2)
        # [b, n_c, n_heads, lq, lc]
        product = self.ff1(product.view(-1, self.lq, self.lc)).view(b, n_c, -1, self.lq, self.lc)
        # [b, n_c, n_heads, lq, lc]
        weights = torch.softmax(torch.sum(product, dim=-2), dim=-1)
        # [b, n_c, n_heads, lc]
        w_sum = weights.unsqueeze(3) @ m_proj.unsqueeze(1)
        # [b, n_c, n_heads, 1, lc]
        output = w_sum.squeeze(-2).transpose(-1, -2) @ self.W_0
        # [b, n_c, lc, 1]

        return torch.softmax(output.squeeze(-1), dim=-1)
        # [b, n_c, lc]


class SelfAttnContxRetrv(torch_nn.Module):
    def __init__(self, lq, lc, n_heads, d_hid, device):
        super().__init__()

        self.lq = lq
        self.lc = lc

        d_k = d_hid // n_heads
        self.W_q = Parameter(torch.rand(n_heads, d_hid, d_k, device=device))
        self.W_c = Parameter(torch.rand(n_heads, d_hid, d_k, device=device))
        self.W_m = Parameter(torch.rand(n_heads, lq, lq, device=device))
        self.W_0 = Parameter(torch.rand(n_heads, 1, device=device))
        self.ff1 = torch_nn.Sequential(
            torch_nn.Linear(self.lq, self.lq),
            torch_nn.Tanh(),
            torch_nn.BatchNorm1d(lc),
        )

    def forward(self, q, c, m):
        # q: [b, lq, d_hid]
        # c: [b, n_c, lc, d_hid]
        # m: [lq, lq]

        b, n_c, _, _ = c.size()

        q_proj = q.unsqueeze(1) @ self.W_q
        # [b, n_heads, lq, d_k]
        c_proj = c.unsqueeze(2) @ self.W_c
        # [b, n_c, n_heads, lc, d_k]
        m_proj = m.view(1, 1, self.lq, self.lq) @ self.W_m
        # [b, n_heads, lq, lq]

        product = c_proj @ q_proj.unsqueeze(1).transpose(-1, -2)
        # [b, n_c, n_heads, lc, lq]
        product = self.ff1(product.view(-1, self.lc, self.lq)).view(b, n_c, -1, self.lc, self.lq)
        # [b, n_c, n_heads, lc, lq]
        weights = torch.softmax(torch.sum(product, dim=-2), dim=-1)
        # [b, n_c, n_heads, lq]
        w_sum = weights.unsqueeze(3) @ m_proj.unsqueeze(1)
        # [b, n_c, n_heads, 1, lq]
        output = w_sum.squeeze(-2).transpose(-1, -2) @ self.W_0
        # [b, n_c, lq, 1]

        return torch.softmax(output.squeeze(-1).sum(1), dim=-1)
        # [b, lq]


class Reasoning(torch_nn.Module):
    def __init__(self, lq, lc, n_heads, d_hid, dropout, device):
        super().__init__()

        self.query_retrv = SelfAttnQueryRetrv(
            lq=lq, lc=lc, n_heads=n_heads, d_hid=d_hid, device=device
        )
        self.mem_c_retrv = Parameter(torch.rand(lc, lc))
        self.contx_retrv = SelfAttnContxRetrv(
            lq=lq, lc=lc, n_heads=n_heads, d_hid=d_hid, device=device
        )
        self.mem_q_retrv = Parameter(torch.rand(lq, lq))

        self.ff1 = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid * 2),
            torch_nn.Tanh(),
            torch_nn.Dropout(dropout),
        )
        self.lin1 = torch_nn.Linear(2 * d_hid, 2 * d_hid)

    def forward(self, q, c):
        # q: [b, lq, d_hid]
        # c: [b, n_c, lc, d_hid]

        b, n_c, _, d_hid = c.size()

        ## Retrieve filters from long-term memory and apply them
        c_filter = self.query_retrv(q=q, c=c, m=self.mem_c_retrv)
        # [b, n_c, lc]
        q_filter = self.contx_retrv(q=q, c=c, m=self.mem_q_retrv)
        # [b, lq]
        c = c * c_filter.unsqueeze(-1)
        # [b, n_c, lc, d_hid]
        q = q * q_filter.unsqueeze(-1)
        # [b, lq, d_hid]

        ## Use CoAttention to sequentially update context
        c_ = []
        for i in range(n_c):
            c_i = c[:, i]

            A = self.ff1(q) @ self.lin1(self.ff1(c_i)).transpose(-1, -2)
            # [b, lq, lc]

            q = torch.softmax(A, dim=-1) @ c_i
            # [b, lq, d_hid]
            c_i = torch.softmax(A, dim=-2).transpose(-1, -2) @ q
            # [b, lc, d_hid]

            c_.append(c_i.unsqueeze(1))

        Y = torch.cat(c_, dim=1).view(b, -1, d_hid)
        # [b, n_c*lc, d_hid]

        return Y
