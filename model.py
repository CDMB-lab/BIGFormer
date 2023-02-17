# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model.py
   Description :
   Author :       zouqi
   date：          2023/2/17
-------------------------------------------------
   Change Activity:
                   2023/2/17:
-------------------------------------------------
"""
__author__ = 'zouqi'

from torch import nn

from layers import DenseGCNConv, PLSA, GRIC


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, norm=False, shortcut=True):
        super(MLP, self).__init__()
        self.fcs = nn.ModuleList()
        self.shortcut = shortcut
        for idx in range(num_layers):
            in_ = in_dim if idx == 0 else hidden_dim // (2 ** (idx - 1))
            out_ = hidden_dim // (2 ** idx) if idx != num_layers - 1 else out_dim

            self.fcs.append(nn.Linear(in_, out_))
            if norm:
                self.fcs.append(nn.BatchNorm1d(out_))
            self.fcs.append(nn.ReLU())
        if self.shortcut:
            self.linear_shortcut = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x_ = self.fcs[0](x)
        for idx in range(1, len(self.fcs)):
            x_ = self.fcs[idx](x_)
        if self.shortcut:
            return x_ + self.linear_shortcut(x)
        else:
            return x_


class GMLP(nn.Module):
    def __init__(self, in_dim, in_units, num_layers, num_targets):
        super(GMLP, self).__init__()

        self.encoder = MLP(in_dim * in_units, in_dim * in_units, num_targets, num_layers, norm=True, shortcut=True)

        self.decoder = nn.Softmax()

    def forward(self, data):
        x, adj = data.x, data.adj
        x = self.encoder(x.view(x.shape[0], -1))
        return self.encoder(x)


class GCN(nn.Module):
    def __init__(self, in_dim, in_units, num_layers, num_targets):
        super(GCN, self).__init__()

        self.encoder = nn.ModuleList()
        for idx in range(num_layers):
            self.encoder.append(DenseGCNConv(in_dim, in_dim))

        self.decoder = nn.ModuleDict({
            'CH': MLP(in_dim * in_units, in_dim * in_units, num_targets, 2, norm=True, shortcut=False),
            'softmax': nn.Softmax()
        })

    def forward(self, data):
        x, adj = data.x, data.adj

        for idx in range(len(self.encoder)):
            x = self.encoder[idx](x, adj).relu()

        x = x.view(x.shape[0], -1)
        y_ = self.decoder['CH'](x)
        return self.decoder['softmax'](y_)


class PLSA_G(nn.Module):
    def __init__(self, in_dim, in_units, num_layers, num_targets):
        super(PLSA_G, self).__init__()

        self.encoder = nn.ModuleList()
        for idx in range(num_layers):
            self.encoder.append(PLSA(in_dim, in_dim))

        self.decoder = nn.ModuleDict({
            'CH': MLP(in_dim * in_units, in_dim * in_units, num_targets, 2, norm=True, shortcut=False),
            'softmax': nn.Softmax()
        })

    def forward(self, data):
        x, adj = data.x, data.adj

        for idx in range(len(self.encoder)):
            x = self.encoder[idx](x, adj).relu()

        x = x.view(x.shape[0], -1)
        y_ = self.decoder['CH'](x)
        return self.decoder['softmax'](y_)


class BIGFormer(nn.Module):
    def __init__(self, in_dim, in_units, num_heads,
                 num_targets, conv, norm_type, attention_type, num_PLSA):
        super(BIGFormer, self).__init__()
        self.attention_type = attention_type
        self.num_PLSA = num_PLSA

        self.encoder = nn.ModuleDict({
            'PLSAs': nn.ModuleList([PLSA(in_dim, in_dim) for _ in range(num_PLSA)]),
            'GRIC': GRIC(dim_Q=in_dim, dim_K=in_dim, dim_V=in_dim, num_nodes=in_units,
                         num_heads=num_heads, norm_type=norm_type, conv=conv,
                         attention_type=attention_type)
        })

        self.decoder = nn.ModuleDict({
            'CH': MLP(in_dim * in_units, in_dim * in_units, num_targets, 2, norm=True, shortcut=False),
            'softmax': nn.Softmax()
        })

    def forward(self, data):
        x, adj = data.x, data.adj

        if self.num_PLSA > 0:
            for idx in range(self.num_PLSA):
                x = self.encoder['PLSAs'][idx](x, adj).relu()

        entanglement = x
        collaboration = self.encoder['GRIC'](x, adj)

        G_ = entanglement + collaboration

        y_ = self.decoder['CH'](G_.view(x.shape[0], -1))
        return self.decoder['softmax'](y_)

