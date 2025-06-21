# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
from dataclasses import dataclass
from logging import getLogger
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...kernel_backend import KernelBackend
from .triton_implementation import embedding_bag_cute


class EmbeddingBag(nn.EmbeddingBag):
    def forward(
        self, indices: torch.Tensor, scores: torch.Tensor, kernel_backend: KernelBackend = KernelBackend.triton
    ) -> torch.Tensor:
        if kernel_backend == KernelBackend.triton:
            output = embedding_bag_cute(input=indices, weight=self.weight, per_sample_weights=scores, mode="sum")
        elif kernel_backend == KernelBackend.torch:
            output = F.embedding_bag(input=indices, weight=self.weight, per_sample_weights=scores, mode="sum")
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        return output

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)


logger = getLogger()


@dataclass
class ProductKeyArgs:
    is_enabled: bool = False
    layers: str = ""  # Which layers to have the memory with product key on (example "6,12")
    mem_n_keys: int = 1024  # The number of keys in the memory.
    mem_heads: int = 4  # Number of memory reading heads
    mem_knn: int = 32  # Number of memory slots to read / update - k-NN to the query
    mem_share_values: bool = True  # Share values across memories
    mem_k_dim: int = 512  # Memory keys dimension
    mem_v_dim: int = -1  # Memory values dimension (-1 for automatic output dimension)
    swilu_projection: bool = True
    value_fixed_lr: Optional[float] = 0.001  # The learning rate for the value of the PK network
    mem_gated: bool = False
    peer_variant: bool = False


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class HashingMemory(nn.Module):

    VALUES = None
    EVAL_MEMORY = True

    def __init__(
        self,
        input_dim,
        output_dim,
        value_fixed_lr=0.001,  # added for xlformers and replaces mem_value_optimizer, set to None to use the same learning rate as the rest of the model
        # global parameters
        mem_k_dim=512,  # Memory keys dimension
        mem_v_dim=-1,  # Memory values dimension (-1 for automatic output dimension)
        mem_heads=4,  # Number of memory reading heads
        mem_knn=32,  # Number of memory slots to read / update - k-NN to the query
        mem_share_values=True,  # Share values across memories
        # keys
        mem_n_keys=1024,  # Number of keys
        # queries
        mem_query_bias=True,  # Query MLP bias
        mem_query_batchnorm=False,  # Query MLP batch norm
        # gating
        mem_gated=False,  # gated memory
        # values initialization
        # dropout
        mem_input_dropout=0.0,  # Input dropout
        mem_query_dropout=0.0,  # Query dropout
        mem_value_dropout=0.0,  # Value dropout
        # architecture
        peer_variant=False,  # Replaces the PK memory with the PEER variant (Parameter Efficient Expert Retrieval)
        swilu_projection=True,
    ):
        # Check parameters
        # even number of key dimensions for product quantization
        assert mem_k_dim >= 2

        # dropout
        assert 0 <= mem_input_dropout < 1
        assert 0 <= mem_query_dropout < 1
        assert 0 <= mem_value_dropout < 1

        # PEER variant
        assert not (
            peer_variant and mem_v_dim > 0
        ), f"Cannot use PEER variant with a value dimension different from the input dimension (mem_v_dim=-1)"

        # dimensions
        assert mem_k_dim % 2 == 0
        assert mem_heads >= 2

        # query batchnorm
        if mem_query_batchnorm:
            logger.warning(
                "WARNING: if you use batch normalization, be sure that you use batches of sentences with the same size at training time. Otherwise, the padding token will result in incorrect mean/variance estimations in the BatchNorm layer."
            )

        # initialize
        super().__init__()
        self.use_peer_variant = peer_variant

        # global parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        # number of indices / entries in the memory
        self.size = mem_n_keys**2
        self.k_dim = mem_k_dim

        self.v_dim = mem_v_dim if mem_v_dim > 0 else output_dim

        # values initialization
        self.swilu_proj = swilu_projection
        self.v_proj = mem_v_dim > 0 or self.swilu_proj
        self.heads = mem_heads
        self.knn = mem_knn

        # dropout
        self.input_dropout = mem_input_dropout
        self.query_dropout = mem_query_dropout
        self.value_dropout = mem_value_dropout

        # initialize keys
        self.keys = nn.Parameter(torch.empty(2 * self.heads * int(self.size**0.5), self.k_dim // 2))

        # optionally use the same values for all memories
        self.mem_share_values = mem_share_values

        self.original = not self.mem_share_values or HashingMemory.VALUES is None

        # initialize the values
        if self.original:
            if not self.use_peer_variant:  # PK
                self.values = EmbeddingBag(self.size, self.v_dim)
                HashingMemory.VALUES = self.values
            else:  # PEER
                self.values_u = nn.Embedding(self.size, self.v_dim)
                self.values_v = nn.Embedding(self.size, self.v_dim)
                HashingMemory.VALUES = self.values_u, self.values_v
        else:
            if not self.use_peer_variant:  # PK
                self.values = None
            else:  # PEER
                self.values_u = None
                self.values_v = None
        self.value_fixed_lr = value_fixed_lr

        if self.v_proj:
            proj_input = mem_v_dim
            if self.swilu_proj and proj_input < 0:
                proj_input = output_dim
            self.value_proj = torch.nn.Linear(proj_input, output_dim)
        if self.swilu_proj:
            self.swilu_projection = torch.nn.Linear(self.input_dim, proj_input)
        # gated memory
        self.gating = None
        if mem_gated:
            self.gating = torch.nn.Linear(input_dim, 1)

        # query network
        # layer sizes / number of features
        l_sizes = (self.input_dim, self.heads * self.k_dim)

        self.query_proj = QueryMLP(
            self.input_dim,
            self.heads,
            self.k_dim,
            l_sizes,
            bias=mem_query_bias,
            batchnorm=mem_query_batchnorm,
        )

    def reset_parameters(self, init_std=None, factor=1.0):
        # keys
        bound = 1 / math.sqrt(self.k_dim)
        nn.init.uniform_(self.keys, a=-bound, b=bound)
        # values
        if self.original:
            if not self.use_peer_variant:
                nn.init.normal_(self.values.weight, mean=0, std=self.v_dim**-0.5)
            else:
                nn.init.normal_(self.values_u.weight, mean=0, std=self.v_dim**-0.5)
                nn.init.normal_(self.values_v.weight, mean=0, std=self.v_dim**-0.5)
        # queries
        nn.init.xavier_uniform_(self.query_proj.query_mlps[0].weight)
        # value projection
        if self.v_proj:
            nn.init.normal_(self.value_proj.weight, mean=0, std=self.output_dim**-0.5)
        if self.swilu_proj:
            nn.init.normal_(self.swilu_projection.weight, mean=0, std=self.output_dim**-0.5)
        # fixed learning rate:
        if self.original:
            if self.use_peer_variant:
                for p in self.values_u.parameters():
                    p.fixed_lr = self.value_fixed_lr
                    p.pk_value_param = True
                for p in self.values_v.parameters():
                    p.fixed_lr = self.value_fixed_lr
                    p.pk_value_param = True
            else:
                for p in self.values.parameters():
                    p.fixed_lr = self.value_fixed_lr
                    p.pk_value_param = True
        if self.gating is not None:
            nn.init.normal_(self.gating.weight, mean=0, std=self.input_dim**-0.5)

    def forward(self, input):
        """
        Read from the memory.
        """
        B, T, C = input.shape
        input = input.view(-1, self.input_dim)

        # input dimensions
        assert input.shape[-1] == self.input_dim
        prefix_shape = input.shape[:-1]

        # compute query / store it
        bs = np.prod(prefix_shape)
        input = F.dropout(input, p=self.input_dropout, training=self.training)  # input shape
        query = self.query_proj(input)  # (bs * heads, k_dim)
        query = F.dropout(query, p=self.query_dropout, training=self.training)  # (bs * heads, k_dim)
        assert query.shape == (bs * self.heads, self.k_dim)

        # get indices
        knn = self.knn
        scores, indices = self.get_indices(query, knn)  # (bs * heads, knn) ** 2

        # store indices / scores (eval mode only - for usage statistics)
        if not self.training and HashingMemory.EVAL_MEMORY:
            self.last_indices = indices.view(bs, self.heads, knn).detach().cpu()
            self.last_scores = scores.view(bs, self.heads, knn).detach().cpu().float()

        # re-scoring
        scores = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs * heads, knn)

        # merge heads / knn (since we sum heads)
        indices = indices.view(bs, self.heads * knn)  # (bs, heads * knn)
        scores = scores.view(bs, self.heads * knn)  # (bs, heads * knn)

        if not self.use_peer_variant:
            output = self.values(indices, scores)  # (bs, v_dim)
            if self.v_proj and not self.swilu_proj:
                output = self.value_proj(output)
            if self.swilu_proj:
                output = self.value_proj(output * F.silu(self.swilu_projection(input)))
        else:
            u = self.values_u(indices)
            x = torch.einsum("bh, blh->bl", input, u)  # (bs, v_dim) , (bs, heads * knn, v_dim) -> (bs, heads * knn)
            x = F.gelu(x)  # This can be either GeLU or ReLU
            v = self.values_v(indices)
            x = x * scores  # (bs, heads * knn)
            output = torch.einsum("bl, blh->bh", x, v)  # (bs, heads * knn) , (bs, heads * knn, v_dim) -> (bs, v_dim)

        output = F.dropout(output, p=self.value_dropout, training=self.training)  # (bs, v_dim)

        # reshape output
        if len(prefix_shape) >= 2:
            output = output.view(prefix_shape + (self.v_dim,))  # (..., v_dim)

        if self.gating:
            output = F.sigmoid(self.gating(input)) * output
        output = output.view(B, T, -1)
        return output

    def get_indices(self, query, knn):
        assert query.dim() == 2 and query.size(1) == self.k_dim
        bs = len(query) // self.heads
        query = query.view(-1, self.heads, self.k_dim)
        half = self.k_dim // 2
        # keys : (heads, 2, n_keys, half)
        # keys1 : (heads, n_keys, half)
        keys = self.keys.view(self.heads, 2, -1, half)
        keys1 = keys[:, 0, :, :]
        keys2 = keys[:, 1, :, :]
        n_keys = len(keys[0][0])

        # split query for product quantization
        q1 = query[:, :, :half]  # (bs, heads, half)
        q2 = query[:, :, half:]  # (bs, heads, half)

        # compute indices with associated scores
        scores1 = torch.einsum("blh, lkh->blk", q1, keys1)  # (bs , heads, n_keys ** 0,5)
        scores2 = torch.einsum("blh, lkh->blk", q2, keys2)  # (bs , heads, n_keys ** 0,5)

        scores1, indices1 = scores1.topk(knn, dim=2, largest=True)  # (bs, heads, knn)
        scores2, indices2 = scores2.topk(knn, dim=2, largest=True)  # (bs, heads, knn)

        # cartesian product on best candidate keys
        all_scores = (
            scores1.view(bs, self.heads, knn, 1).expand(bs, self.heads, knn, knn)
            + scores2.view(bs, self.heads, 1, knn).expand(bs, self.heads, knn, knn)
        ).view(
            bs, self.heads, -1
        )  # (bs, heads, knn ** 2)
        all_indices = (
            indices1.view(bs, self.heads, knn, 1).expand(bs, self.heads, knn, knn) * n_keys
            + indices2.view(bs, self.heads, 1, knn).expand(bs, self.heads, knn, knn)
        ).view(
            bs, self.heads, -1
        )  # (bs, heads, knn ** 2)

        # select overall best scores and indices
        scores, best_indices = torch.topk(all_scores, k=knn, dim=2, largest=True, sorted=True)  # (bs, heads, knn)
        indices = all_indices.gather(2, best_indices)  # (bs, knn)

        # return scores with indices
        assert scores.shape == indices.shape == (bs, self.heads, knn)
        return scores.view(bs * self.heads, knn), indices.view(bs * self.heads, knn)


class QueryMLP(nn.Module):
    def __init__(self, input_dim, heads, k_dim, sizes, bias=False, batchnorm=False):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.k_dim = k_dim
        self.sizes = sizes
        assert sizes[0] == input_dim
        assert sizes[-1] == (heads * k_dim)

        # MLPs
        sizes_ = list(sizes)
        sizes_[-1] = sizes_[-1]
        self.query_mlps = QueryMLP.mlp(sizes_, bias=bias, batchnorm=batchnorm)

    @staticmethod
    def mlp(sizes, bias=True, batchnorm=True):
        """
        Generate a feedforward neural network.
        """
        assert len(sizes) >= 2
        pairs = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        layers = []

        for i, (dim_in, dim_out) in enumerate(pairs):
            layers.append(nn.Linear(dim_in, dim_out, bias=bias))
            if batchnorm:
                layers.append(nn.BatchNorm1d(dim_out))
            if i < len(pairs) - 1:
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() > 2:
            input = input.reshape(-1, self.input_dim)

        B = input.size(0)

        outputs = [m(input) for m in self.query_mlps]
        query = torch.cat(outputs, 1) if len(outputs) > 1 else outputs[0]

        assert query.size() == (B, self.heads * self.k_dim)
        return query.view(B * self.heads, self.k_dim)
