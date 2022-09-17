import torch
import torch.nn.functional as F
from torch import nn, einsum


class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()

        self.layers = nn.Sequential()

        for d in range(len(dims) - 2):
            self.layers.add_module(f'layer-{d}', nn.Linear(dims[d], dims[d + 1], bias=True))
            self.layers.add_module(f'layer-act-{d}', nn.ReLU())

        self.layers.add_module('layer-final}', nn.Linear(dims[-2], dims[-1]))

    def forward(self, x):
        if len(x.shape) == 1:
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
        super().__init__()  # init the base class

    def forward(self, input):
        return quadratic(input)


class NumericalEmbedding(nn.Module):
    def __init__(self, num_continuous, dim, num_high_dim):
        super().__init__()
        self.num_continuous = num_continuous
        self.dim = dim
        self.simple_MLP = nn.ModuleList([MLP([1, num_high_dim, dim]) for _ in
            range(self.num_continuous)])

    def forward(self, x_cont):
        n1, n2 = x_cont.shape
        x_cont_enc = torch.empty(n1, n2, self.dim)
        for i in range(self.num_continuous):
            x_cont_enc[:, i, :] = self.simple_MLP[i](x_cont[:, i])

        return x_cont_enc


class CategoricalEmbedding(nn.Module):
    def __init__(self, categories, dim):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        self.num_unique_categories = sum(categories)
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0))
        self.categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.embeds = nn.Embedding(self.num_unique_categories, dim)

    def forward(self, x_categ):
        x_categ = x_categ + self.categories_offset.type_as(x_categ)
        x_categ_enc = self.embeds(x_categ)
        return x_categ_enc


class EnsembleActivations(nn.Module):
    def __init__(self, act1, act2):
        super().__init__()
        self.act1 = act1
        self.act2 = act2

    def forward(self, x):
        b, f = x.size()
        mid = f // 2

        x_start = x[:, :mid]
        x_start = self.act1(x_start)

        x_end = x[:, mid:]
        x_end = self.act2(x_end)

        x = torch.cat([x_start, x_end], dim=1)
        return x


class FiFa(nn.Module):
    def __init__(
            self,
            *,
            categories,  # a list of categories within each categorical field
            num_continuous,
            embedding_dim,
            y_dim=2,
            final_dim_factor=1,
            dropout=0.0,
            num_dim=100,
            num_dim_factor=0,
            act='gelu'
    ):
        super().__init__()

        self.num_categories = len(categories)
        self.num_continuous = num_continuous

        input_size = (embedding_dim * (self.num_categories)) + (embedding_dim * self.num_continuous)
        self.embedding_dim = embedding_dim
        self.high_dim = int(round(final_dim_factor * input_size))
        self.out_dim = y_dim if y_dim > 2 else 1

        self.categorical_embedding = CategoricalEmbedding(categories, embedding_dim)
        self.numerical_embedding = NumericalEmbedding(num_continuous, embedding_dim, num_dim_factor * input_size if (num_dim_factor > 0) else num_dim)

        self.linear = nn.Linear(input_size, self.out_dim, bias=True)
        self.U = nn.Linear(input_size, self.high_dim, bias=True)
        self.W_out = nn.Linear(self.high_dim, self.out_dim, bias=False)

        if act == 'relu':
            self.activation = nn.ReLU()
        elif act == 'quad':
            self.activation = Quadratic()
        elif act == 'quad-relu':
            self.activation = EnsembleActivations(Quadratic(), nn.ReLU())
        else:
            self.activation = nn.GELU()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_categ, x_cont):
        x_categ_enc = self.categorical_embedding(x_categ)
        x_cont_enc = self.numerical_embedding(x_cont)

        if x_cont_enc is not None and x_cont_enc is not None:
            x = torch.cat((x_categ_enc, x_cont_enc), dim=1)
        elif x_cont_enc is not None:
            x = x_cont_enc
        else:
            x = x_categ_enc

        x = x.flatten(start_dim=1)
        x_linear = self.linear(x)

        x = self.dropout(x)
        x = self.U(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.W_out(x)

        x_final = x_linear + x
        return x_final

