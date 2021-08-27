import torch
import torch.nn as torch_nn
from torch.nn.parameter import Parameter


class SelfAttnQueryRetrv(torch_nn.Module):
    def __init__(self, lq, lc, n_heads, d_bert, device):
        super().__init__()

        self.lc = lc
        self.lq = lq

        d_k = d_bert // n_heads
        self.W_q = Parameter(torch.rand(n_heads, d_bert, d_k, device=device))
        self.W_c = Parameter(torch.rand(n_heads, d_bert, d_k, device=device))
        self.W_m = Parameter(torch.rand(n_heads, lc, lc, device=device))
        self.W_0 = Parameter(torch.rand(n_heads, 1, device=device))
        self.ff1 = torch_nn.Sequential(
            torch_nn.Linear(self.lc, self.lc),
            torch_nn.Tanh(),
            torch_nn.BatchNorm1d(lq),
        )

    def forward(self, q, c, m):
        # q: [b, lq, d_bert]
        # c: [b, lc, d_bert]
        # m: [lc, lc]

        b = c.size(0)

        q_proj = q.unsqueeze(1) @ self.W_q
        # [b, n_heads, lq, d_k]
        c_proj = c.unsqueeze(1) @ self.W_c
        # [b, n_heads, lc, d_k]
        m_proj = m.view(1, 1, self.lc, self.lc) @ self.W_m
        # [b, n_heads, lc, lc]

        product = q_proj @ c_proj.transpose(-1, -2)
        # [b, n_heads, lq, lc]
        product = self.ff1(product.view(-1, self.lq, self.lc)).view(b, -1, self.lq, self.lc)
        # [b, n_heads, lq, lc]
        weights = torch.softmax(torch.sum(product, dim=-2), dim=-1)
        # [b, n_heads, lc]
        w_sum = weights.unsqueeze(2) @ m_proj
        # [b, n_heads, 1, lc]
        output = w_sum.squeeze(-2).transpose(-1, -2) @ self.W_0
        # [b, lc, 1]

        return torch.softmax(output.squeeze(-1), dim=-1)
        # [b, lc]


class SelfAttnContxRetrv(torch_nn.Module):
    def __init__(self, lq, lc, n_heads, d_bert, device):
        super().__init__()

        self.lq = lq
        self.lc = lc

        d_k = d_bert // n_heads
        self.W_q = Parameter(torch.rand(n_heads, d_bert, d_k, device=device))
        self.W_c = Parameter(torch.rand(n_heads, d_bert, d_k, device=device))
        self.W_m = Parameter(torch.rand(n_heads, lq, lq, device=device))
        self.W_0 = Parameter(torch.rand(n_heads, 1, device=device))
        self.ff1 = torch_nn.Sequential(
            torch_nn.Linear(self.lq, self.lq),
            torch_nn.Tanh(),
            torch_nn.BatchNorm1d(lc),
        )

    def forward(self, q, c, m):
        # q: [b, lq, d_bert]
        # c: [b, nc, lc, d_bert]
        # m: [lq, lq]

        b, nc, _, _ = c.size()

        q_proj = q.unsqueeze(1) @ self.W_q
        # [b, n_heads, lq, d_k]
        c_proj = c.unsqueeze(2) @ self.W_c
        # [b, nc, n_heads, lc, d_k]
        m_proj = m.view(1, 1, self.lq, self.lq) @ self.W_m
        # [b, n_heads, lq, lq]

        product = c_proj @ q_proj.unsqueeze(1).transpose(-1, -2)
        # [b, nc, n_heads, lc, lq]
        product = self.ff1(product.view(-1, self.lc, self.lq)).view(b, nc, -1, self.lc, self.lq)
        # [b, nc, n_heads, lc, lq]
        weights = torch.softmax(torch.sum(product, dim=-2), dim=-1)
        # [b, nc, n_heads, lq]
        w_sum = weights.unsqueeze(3) @ m_proj.unsqueeze(1)
        # [b, nc, n_heads, 1, lq]
        output = w_sum.squeeze(-2).transpose(-1, -2) @ self.W_0
        # [b, nc, lq, 1]

        return torch.softmax(output.squeeze(-1).sum(1), dim=-1)
        # [b, lq]


class PersistentMemoryCell(torch_nn.Module):
    def __init__(self, lq, lc, n_heads, d_bert, device):
        super().__init__()

        self.query_retrv = SelfAttnQueryRetrv(
            lq=lq, lc=lc, n_heads=n_heads, d_bert=d_bert, device=device
        )
        self.mem_c_retrv = Parameter(torch.rand(lc, lc))
        # self.contx_retrv = SelfAttnContxRetrv(
        #     lq=lq, lc=lc, n_heads=n_heads, d_bert=d_bert, device=device
        # )
        # self.mem_q_retrv = Parameter(torch.rand(lq, lq))

    def forward(self, q, c):
        # q: [b, lq_, d_bert]
        # c: [b, lc_, d_bert]

        ## Retrieve filters from long-term memory and apply them
        c_filter = self.query_retrv(q=q, c=c, m=self.mem_c_retrv)
        # [b, lc_]
        # NOTE: Temporarily not use q_filter
        # q_filter = self.contx_retrv(q=q, c=c, m=self.mem_q_retrv)
        # [b, lq_]

        return c_filter
