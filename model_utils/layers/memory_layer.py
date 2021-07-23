from typing import Any

import torch.nn.functional as torch_f
import torch.nn as torch_nn
import torch

from model_utils.layers.reasoning_layer.utils import (
    MemoryBasedContextRectifier,
    MemoryBasedQuesRectifier,
)


class MemoryBasedReasoning(torch_nn.Module):
    def __init__(
        self,
        l_q2,
        l_c70,
        n_c0,
        n_heads_trans,
        n_layers_trans,
        d_hid,
        device: Any = "cpu",
    ):
        super().__init__()

        self.l_q = l_q
        self.l_c = l_c
        self.n_c = n_c
        self.d_hid = d_hid

        self.ques_rectifier = MemoryBasedQuesRectifier(
            l_q=l_q,
            d_hid=d_hid,
            n_heads_trans=n_heads_trans,
            n_layers_trans=n_layers_trans,
            device=device,
        )

        self.contx_rectifier = MemoryBasedContextRectifier(
            l_c=l_c,
            n_c=n_c,
            d_hid=d_hid,
            n_heads_trans=n_heads_trans,
            n_layers_trans=n_layers_trans,
            device=device,
        )

        self.ff1 = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid),
            torch_nn.Linear(d_hid, d_hid),
            torch_nn.Tanh(),
            torch_nn.BatchNorm1d(l_q),
        )
        self.ff2 = torch_nn.Sequential(
            torch_nn.Linear(d_hid, d_hid),
            torch_nn.Linear(d_hid, d_hid),
            torch_nn.Tanh(),
            torch_nn.BatchNorm1d(l_c),
        )
        self.lin2 = torch_nn.Linear(d_hid, d_hid)
        self.lin3 = torch_nn.Linear(d_hid * 2, d_hid)

    def forward(self, q, c):
        # q: [b, l_q, d_hid]
        # c: [b, n_c, l_c, d_hid]

        b, n_c, l_c, _ = c.size()

        ###################################
        # Rectìfy question and c
        ###################################
        rect_ques = self.ques_rectifier(q=q, c=c)
        # [b, l_q, d_hid]
        rect_context, contx_rectifier = self.contx_rectifier(q=q, c=c)
        # rect_context: [b, n_c, l_c, d_hid]
        # contx_rectifier: [b, n_c, l_c]

        ###################################
        # Use CoAttention to capture
        ###################################
        contx_rectifier = torch.max(contx_rectifier, dim=2)[0]
        # [b, n_c]
        contx_rectifier = torch_f.gumbel_softmax(
            contx_rectifier, tau=1, hard=True, dim=1
        )
        # [b, n_c]
        contx_rectifier = contx_rectifier.view(b, n_c, 1, 1).repeat(
            1, 1, l_c, self.d_hid * 2
        )

        paras = []
        for ith in range(n_c):
            para = rect_context[:, ith]
            # [b, l_c, d_hid]

            A_i = torch.bmm(
                self.ff1(rect_ques), self.lin2(self.ff2(para)).transpose(1, 2)
            )
            # [b, l_q, l_c]

            A_q = torch.softmax(A_i, dim=1)
            A_d = torch.softmax(A_i, dim=2)

            C_q = torch.bmm(A_q, para)
            # [b, l_q, d_hid]

            para = torch.bmm(A_d.transpose(1, 2), torch.cat((C_q, rect_ques), dim=-1))
            # [b, l_c, d_hid * 2]

            paras.append(para.unsqueeze(1))

        paras = torch.cat(paras, dim=1)
        # [b, n_c, l_c, d_hid * 2]

        Y = (contx_rectifier * paras).sum(1)
        # [b, l_c, d_hid * 2]

        Y = self.lin3(Y)
        # [b, l_c, d_hid]

        return Y
