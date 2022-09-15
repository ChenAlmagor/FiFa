import torch
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np
from einops import rearrange


class simple_MLP(nn.Module):
    def __init__(self,dims):
        super(simple_MLP, self).__init__()

        self.layers = nn.Sequential()

        for d in range(len(dims)-2):
            self.layers.add_module(f'layer-{d}', nn.Linear(dims[d], dims[d+1], bias=True))
            self.layers.add_module(f'layer-act-{d}', nn.ReLU())

        self.layers.add_module('layer-final}', nn.Linear(dims[-2], dims[-1]))

    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x



def quadratic(input):
    '''
    Applies the quadratic function element-wise:
        quadratic(x) = x ** 2
    '''
    return torch.pow(input, 2)

class Quadratic(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        return quadratic(input)

class MF(nn.Module):
    def __init__(self, embedding_cat_dim, dim, act='gelu', out_dim=1, dropout=0.0):
        super().__init__()
        self.U = nn.Linear(embedding_cat_dim, dim, bias=True)
        self.dropout = nn.Dropout(p=dropout)

        if act == 'relu':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        elif act == 'quad':
            self.act1 = Quadratic()
            self.act2 = Quadratic()
        elif act == 'half':
            self.act1 = Quadratic()
            self.act2 = nn.ReLU()
        else:
            self.act1 = nn.GELU()
            self.act2 = nn.GELU()

        self.diag = nn.Linear(dim, out_dim, bias=False)


    def forward(self, x):

        x = self.dropout(x)
        x = self.U(x)

        b, f = x.size()
        mid = f // 2

        x_start = x[:, :mid]
        x_start = self.act1(x_start)

        x_end = x[:, mid:]
        x_end = self.act2(x_end)

        x = torch.cat([x_start, x_end], dim=1)
        x = self.dropout(x)

        x = self.diag(x)
        return x


class PostEmbFactorization(nn.Module):
    def __init__(self, emb_cat_dim, dim, out_dim, dropout=0.0, y_index=None, act='gelu'):
        super().__init__()
        self.emb_cat_dim = emb_cat_dim
        self.dim = dim

        self.linear = nn.Linear(self.emb_cat_dim, out_dim, bias=True)
        self.mf = MF(self.emb_cat_dim, self.dim, act=act, dropout=dropout)
        self.y_index=y_index


    def forward(self, x, x_cont=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)

        if self.y_index is not None:
            b, f, e = x.size()
            idx = torch.arange(f)
            idx_no_y = idx != self.y_index
            x = x[:, idx_no_y, ]

        x = x.flatten(start_dim=1)
        x_linear = self.linear(x)
        x_mf = self.mf(x)
        x_rep = x_linear + x_mf

        return x_rep


class FiFa(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            num_special_tokens=0,
            y_dim=2,
            final_dim_factor=1,
            dropout=0.0,
            num_dim=100,
            num_dim_factor=0,
            include_y=False,
            act='gelu'
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('categories_offset', categories_offset)

        self.num_continuous = num_continuous
        self.dim = dim

        input_size = (dim * (self.num_categories - 1)) + (dim * num_continuous)

        self.num_dim_factor = num_dim_factor
        self.input_size = input_size
        self.num_dim = num_dim

        self.simple_MLP = nn.ModuleList([simple_MLP(
            [1, num_dim_factor * input_size if (num_dim_factor > 0) else num_dim,
             self.dim]) for _ in range(self.num_continuous)])
        nfeats = self.num_categories + num_continuous

        self.embeds = nn.Embedding(self.total_tokens, self.dim)  # .to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0)
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0)
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, self.dim)


        self.post_emb_factorization = PostEmbFactorization(input_size,
                                                           int(round(final_dim_factor * input_size)),
                                                           out_dim=y_dim if y_dim > 2 else 1,
                                                           dropout=dropout,
                                                           y_index=self.num_categories - 1 if include_y == False else None,
                                                           act=act
                                                           )

    def forward(self, x_categ, x_cont):
        return self.post_emb_factorization(x_categ, x_cont)