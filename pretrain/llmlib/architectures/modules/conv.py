import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Union

class TransposeLayer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        return x

class ConvLayer(nn.Module):
    """
    A 1D GPN Conv layer
    """
    def __init__(
        self,
        hidden_size=None,
        activation = "GELU", #nn.GELU(),
        norm = "LayerNorm",
        **kwargs,
    ):
        super().__init__()
        # TODO deprecate str ctor once hydra integrated.
        # come up with a better way to specify this
        if isinstance(activation, str):
            if activation == "GELU": 
                activation = nn.GELU
            else:
                raise NotImplementedError
        if isinstance(norm, str):
            if norm == "LayerNorm":
                norm = nn.LayerNorm
            else:
                raise NotImplementedError
        self.conv = nn.Sequential(
            TransposeLayer(),
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                padding="same",
                **kwargs,
            ),
            TransposeLayer(),
            activation(),
            norm(hidden_size),
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            activation(),
            norm(hidden_size),
        )

    def forward(self, x) -> torch.Tensor:
        x = x + self.conv(x)
        x = x + self.ffn(x)
        return x


class OneHotEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x) -> torch.Tensor:
        return F.one_hot(x, num_classes=self.hidden_size).float()


class GPNEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size=None,
        n_aux_features=None,
        hidden_size=None,
    ):
        super().__init__()
        assert vocab_size + n_aux_features <= hidden_size
        self.vocab_size = vocab_size
        self.n_aux_features = n_aux_features
        self.hidden_size = hidden_size

    def forward(self, input_ids, aux_features=None) -> torch.Tensor:
        res = F.one_hot(input_ids, num_classes=self.hidden_size).float()
        if aux_features is not None:
            res[:, :, self.vocab_size:self.vocab_size+self.n_aux_features] = aux_features
        return res


def get_dilation_schedule(config):
    return [
        min(config.dilation_max, 2**((i%config.dilation_cycle)//config.dilation_double_every))
        for i in range(config.n_layers)
    ]
