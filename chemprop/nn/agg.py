from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from chemprop.data import FPPoolBatch
from chemprop.nn.hparams import HasHParams
from chemprop.utils import ClassRegistry

__all__ = [
    "Aggregation",
    "AggregationRegistry",
    "MeanAggregation",
    "SumAggregation",
    "NormAggregation",
    "FPPoolAggregation",
    "AttentiveAggregation",
]


class Aggregation(nn.Module, HasHParams):
    """An :class:`Aggregation` aggregates the node-level representations of a batch of graphs into
    a batch of graph-level representations

    .. note::
        this class is abstract and cannot be instantiated.

    See also
    --------
    :class:`~chemprop.v2.models.modules.agg.MeanAggregation`
    :class:`~chemprop.v2.models.modules.agg.SumAggregation`
    :class:`~chemprop.v2.models.modules.agg.NormAggregation`
    """

    def __init__(self, dim: int = 0, *args, **kwargs):
        super().__init__()

        self.dim = dim
        self.hparams = {"dim": dim, "cls": self.__class__}

    @abstractmethod
    def forward(self, H: Tensor, batch: Tensor, fppool_batch: FPPoolBatch | None = None) -> Tensor:
        """Aggregate the graph-level representations of a batch of graphs into their respective
        global representations

        NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
        a zero vector of length `d` in the final output.

        Parameters
        ----------
        H : Tensor
            a tensor of shape ``V x d`` containing the batched node-level representations of ``b``
            graphs
        batch : Tensor
            a tensor of shape ``V`` containing the index of the graph a given vertex corresponds to
        fppool_batch : FPPoolBatch | None, default=None
            optional structured FPPool metadata aligned to the batched atom ordering. Non-FPPool
            aggregations ignore this argument.

        Returns
        -------
        Tensor
            a tensor of shape ``b x d`` containing the graph-level representations
        """


AggregationRegistry = ClassRegistry[Aggregation]()


@AggregationRegistry.register("mean")
class MeanAggregation(Aggregation):
    r"""Average the graph-level representation:

    .. math::
        \mathbf h = \frac{1}{|V|} \sum_{v \in V} \mathbf h_v
    """

    def forward(self, H: Tensor, batch: Tensor, fppool_batch: FPPoolBatch | None = None) -> Tensor:
        index_torch = batch.unsqueeze(1).repeat(1, H.shape[1])
        dim_size = batch.max().int() + 1
        return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, index_torch, H, reduce="mean", include_self=False
        )


@AggregationRegistry.register("sum")
class SumAggregation(Aggregation):
    r"""Sum the graph-level representation:

    .. math::
        \mathbf h = \sum_{v \in V} \mathbf h_v

    """

    def forward(self, H: Tensor, batch: Tensor, fppool_batch: FPPoolBatch | None = None) -> Tensor:
        index_torch = batch.unsqueeze(1).repeat(1, H.shape[1])
        dim_size = batch.max().int() + 1
        return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, index_torch, H, reduce="sum", include_self=False
        )


@AggregationRegistry.register("norm")
class NormAggregation(SumAggregation):
    r"""Sum the graph-level representation and divide by a normalization constant:

    .. math::
        \mathbf h = \frac{1}{c} \sum_{v \in V} \mathbf h_v
    """

    def __init__(self, dim: int = 0, *args, norm: float = 100.0, **kwargs):
        super().__init__(dim, **kwargs)

        self.norm = norm
        self.hparams["norm"] = norm

    def forward(self, H: Tensor, batch: Tensor, fppool_batch: FPPoolBatch | None = None) -> Tensor:
        return super().forward(H, batch, fppool_batch) / self.norm


@dataclass(repr=False, eq=False, slots=True)
class FPPoolAttentionState:
    """Private attention payload from the most recent FPPool forward pass."""

    inner_atom_indices: Tensor
    inner_bit_indices: Tensor
    inner_weights: Tensor
    inter_weights: tuple[Tensor, ...]
    inter_masks: tuple[Tensor, ...]
    global_weights: Tensor
    global_mask: Tensor


def _masked_attention_pool(values: Tensor, logits: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Apply masked attention over the second dimension of a dense tensor.

    Rows with no valid entries yield zero pooled outputs and zero attention weights.
    """
    weights = torch.zeros_like(logits)
    if not mask.any():
        return torch.zeros(
            values.shape[0], values.shape[-1], dtype=values.dtype, device=values.device
        ), weights

    valid_rows = mask.any(dim=1)
    if valid_rows.any():
        masked_logits = logits[valid_rows].masked_fill(~mask[valid_rows], float("-inf"))
        valid_weights = torch.softmax(masked_logits, dim=1).masked_fill(~mask[valid_rows], 0.0)
        weights[valid_rows] = valid_weights

    pooled = (weights.unsqueeze(-1) * values).sum(dim=1)
    return pooled, weights


@AggregationRegistry.register("fppool")
class FPPoolAggregation(Aggregation):
    """Hierarchical fingerprint-guided pooling for molecule representations.

    This implementation follows the attention-only FPPool hierarchy used for milestone 1:

    - atom embeddings -> fingerprint-bit embeddings
    - bit embeddings -> fingerprint-family embeddings
    - family embeddings -> molecule embeddings

    The implementation is multi-family-capable through ``fp_family_lengths`` even though the
    current data path only populates Morgan memberships.
    """

    def __init__(self, dim: int = 0, *args, **kwargs):
        super().__init__(dim, *args, **kwargs)

        self.inner_attn = nn.LazyLinear(1)
        self.inter_attn = nn.LazyLinear(1)
        self.global_attn = nn.LazyLinear(1)
        self.hparams.update(
            {
                "inner_attn": self.inner_attn.__class__.__name__,
                "inter_attn": self.inter_attn.__class__.__name__,
                "global_attn": self.global_attn.__class__.__name__,
            }
        )
        self._last_attention_state: FPPoolAttentionState | None = None

    @property
    def last_attention_state(self) -> FPPoolAttentionState | None:
        """The detached attention payload from the most recent forward pass."""
        return self._last_attention_state

    def _inner_pool(
        self, H: Tensor, batch: Tensor, atom_fp: Tensor, num_graphs: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Aggregate atom embeddings into fingerprint-bit embeddings."""
        num_bits = atom_fp.shape[1]
        bit_embeddings = torch.zeros(num_graphs, num_bits, H.shape[1], dtype=H.dtype, device=H.device)
        bit_mask = torch.zeros(num_graphs, num_bits, dtype=torch.bool, device=H.device)

        active_atom_indices, active_bit_indices = atom_fp.nonzero(as_tuple=True)
        if active_atom_indices.numel() == 0:
            return (
                bit_embeddings,
                bit_mask,
                active_atom_indices,
                active_bit_indices,
                torch.zeros(0, dtype=H.dtype, device=H.device),
            )

        group_ids = batch[active_atom_indices] * num_bits + active_bit_indices
        active_logits = self.inner_attn(H[active_atom_indices]).squeeze(-1)

        max_logits = torch.full(
            (num_graphs * num_bits,), float("-inf"), dtype=H.dtype, device=H.device
        )
        max_logits.scatter_reduce_(0, group_ids, active_logits, reduce="amax", include_self=True)

        stabilized_logits = torch.exp(active_logits - max_logits[group_ids])
        normalization = torch.zeros(num_graphs * num_bits, dtype=H.dtype, device=H.device)
        normalization.scatter_add_(0, group_ids, stabilized_logits)
        active_weights = stabilized_logits / normalization[group_ids]

        group_index = group_ids.unsqueeze(1).expand(-1, H.shape[1])
        flat_embeddings = torch.zeros(
            num_graphs * num_bits, H.shape[1], dtype=H.dtype, device=H.device
        )
        flat_embeddings.scatter_add_(0, group_index, active_weights.unsqueeze(1) * H[active_atom_indices])
        bit_embeddings = flat_embeddings.view(num_graphs, num_bits, H.shape[1])
        bit_mask.view(-1)[group_ids] = True

        return bit_embeddings, bit_mask, active_atom_indices, active_bit_indices, active_weights

    def forward(self, H: Tensor, batch: Tensor, fppool_batch: FPPoolBatch | None = None) -> Tensor:
        if fppool_batch is None:
            raise ValueError("FPPool aggregation requires an FPPoolBatch in the input batch.")

        if fppool_batch.atom_fp.shape[0] != H.shape[0]:
            raise ValueError(
                "FPPool atom membership rows must match the number of atom embeddings in the batch."
            )

        fppool_batch = fppool_batch.to(H.device)
        num_graphs = int(fppool_batch.molecule_atom_slices.numel() - 1)
        if num_graphs == 0:
            return torch.zeros(0, H.shape[1], dtype=H.dtype, device=H.device)

        atom_fp = fppool_batch.atom_fp
        bit_embeddings, bit_mask, active_atom_indices, active_bit_indices, active_weights = self._inner_pool(
            H, batch, atom_fp, num_graphs
        )

        family_embeddings = []
        family_masks = []
        inter_weights = []
        inter_masks = []
        bit_start = 0
        for family_length in fppool_batch.fp_family_lengths.tolist():
            bit_stop = bit_start + family_length
            family_bit_embeddings = bit_embeddings[:, bit_start:bit_stop]
            family_bit_mask = bit_mask[:, bit_start:bit_stop]
            family_logits = self.inter_attn(family_bit_embeddings).squeeze(-1)
            pooled_family, family_weights = _masked_attention_pool(
                family_bit_embeddings, family_logits, family_bit_mask
            )
            family_embeddings.append(pooled_family)
            family_masks.append(family_bit_mask.any(dim=1))
            inter_weights.append(family_weights.detach())
            inter_masks.append(family_bit_mask.detach())
            bit_start = bit_stop

        family_embedding_tensor = torch.stack(family_embeddings, dim=1)
        family_mask_tensor = torch.stack(family_masks, dim=1)
        global_logits = self.global_attn(family_embedding_tensor).squeeze(-1)
        pooled, global_weights = _masked_attention_pool(
            family_embedding_tensor, global_logits, family_mask_tensor
        )

        self._last_attention_state = FPPoolAttentionState(
            inner_atom_indices=active_atom_indices.detach(),
            inner_bit_indices=active_bit_indices.detach(),
            inner_weights=active_weights.detach(),
            inter_weights=tuple(inter_weights),
            inter_masks=tuple(inter_masks),
            global_weights=global_weights.detach(),
            global_mask=family_mask_tensor.detach(),
        )

        return pooled


class AttentiveAggregation(Aggregation):
    def __init__(self, dim: int = 0, *args, output_size: int, **kwargs):
        super().__init__(dim, *args, **kwargs)

        self.hparams["output_size"] = output_size
        self.W = nn.Linear(output_size, 1)

    def forward(self, H: Tensor, batch: Tensor, fppool_batch: FPPoolBatch | None = None) -> Tensor:
        dim_size = batch.max().int() + 1
        attention_logits = self.W(H).exp()
        Z = torch.zeros(dim_size, 1, dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, batch.unsqueeze(1), attention_logits, reduce="sum", include_self=False
        )
        alphas = attention_logits / Z[batch]
        index_torch = batch.unsqueeze(1).repeat(1, H.shape[1])
        return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, index_torch, alphas * H, reduce="sum", include_self=False
        )
