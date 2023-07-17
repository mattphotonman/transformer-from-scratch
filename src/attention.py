import numpy as np
import torch
from torch import nn


class AttentionLayer(nn.Module):

    def __init__(
            self, num_heads: int, dim_input: int, dim_key: int,
            dim_value: int
    ):
        self.query_projector = nn.Parameter(
            torch.rand(num_heads, dim_input, dim_key)
        )
        self.key_projector = nn.Parameter(
            torch.rand(num_heads, dim_input, dim_key)
        )
        self.value_projector = nn.Parameter(
            torch.rand(num_heads, dim_input, dim_value)
        )
        self.output_linear = nn.Parameter(
            torch.rand(num_heads * dim_value, dim_input)
        )

    def forward(
            self, queries: torch.Tensor, keys: torch.Tensor,
            values: torch.Tensor
    ) -> torch.Tensor:
        # queries.size() = (batch, T, dim_input)
        # keys.size() = (batch, T, dim_input)
        # values.size() = (batch, T, dim_input)

        # actually should use torch.tensordot below
        # https://pytorch.org/docs/stable/generated/torch.tensordot.html
        projected_queries = torch.matmul(queries, self.query_projector)
        projected_keys = torch.matmul(keys, self.key_projector)
        projected_values = torch.matmul(values, self.value_projector)

        # projected_queries.size() = (num_heads, T, dim_key)
        # projected_keys.size() = (num_heads, T, dim_key)
        # projected_values.size() = (num_heads, T, dim_value)

        dim_key = projected_queries.size()[2]
        attention_outputs =
