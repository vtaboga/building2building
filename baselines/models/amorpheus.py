"""Amorpheus: type-heterogeneous transformer policy for cross-domain transfer.

Implements the architecture from Section 6.2 of the Building2Building paper.
Each node type in the morphology graph has its own encoder/decoder head,
and a shared transformer aggregates information across all nodes.

The model operates on the morphological universe defined in
:mod:`building2building.morphology`.  Actions are Beta-distributed on
[0, 1] and rescaled to physical units using the local action spaces
carried by each :class:`~building2building.morphology.NodeType`.

Key components:

- :class:`Model` -- shared-weight model with per-NodeType encoder/decoder
  heads.  Call :meth:`Model.condition` to bind to a specific morphology.
- :class:`Policy` -- lightweight handle bound to one morphology; holds
  references to the shared ``Model`` weights.
- :func:`encode_observation` / :func:`decode_action` -- map between the
  flat env interface and per-node local spaces.
- :class:`AmorpheusPolicy` -- standalone ``nn.Module`` wrapper for
  evaluation and checkpointing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from torch.distributions import Distribution

from building2building.morphology import (
    ALL_NODE_TYPES,
    Morphology,
    MorphologyNode,
    NodeType,
)

# ---------------------------------------------------------------------------
# ScaledBeta distribution
# ---------------------------------------------------------------------------


class ScaledBeta(Distribution):
    """Beta distribution rescaled from [0, 1] to [low, high].

    Samples, mean, and entropy are in the physical action space.
    ``log_prob`` accounts for the Jacobian of the affine transform.
    """

    arg_constraints: dict = {}
    has_rsample = True

    def __init__(
        self,
        base: Beta,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> None:
        self.base = base
        self.low = low
        self.high = high
        self.scale = high - low
        super().__init__(base.batch_shape, base.event_shape)

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        return self.low + self.base.sample(sample_shape) * self.scale

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        return self.low + self.base.rsample(sample_shape) * self.scale

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        x01 = (value - self.low) / self.scale
        return self.base.log_prob(x01) - self.scale.log()

    def entropy(self) -> torch.Tensor:
        return self.base.entropy() + self.scale.log()

    @property
    def mean(self) -> torch.Tensor:
        return self.low + self.base.mean * self.scale


# ---------------------------------------------------------------------------
# Building type conditioning
# ---------------------------------------------------------------------------

BUILDING_TYPE_INDEX: dict[str, int] = {
    "Warehouse": 0,
    "SingleFamilyHouse": 1,
    "RetailStandalone": 2,
    "RestaurantFastFood": 3,
    "OfficeMedium": 4,
    "OfficeSmall": 5,
}
N_BUILDING_TYPES = len(BUILDING_TYPE_INDEX)


# ---------------------------------------------------------------------------
# Extensions (ablation toggles)
# ---------------------------------------------------------------------------


@dataclass
class ExtensionConfig:
    """Toggle-able extensions for ablation studies."""

    spectral_pe: bool = False
    spectral_pe_k: int = 8


# ---------------------------------------------------------------------------
# Model (shared weights across all morphologies)
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """Morphology-agnostic model.  One encoder/decoder per NodeType.

    Encoders map ``local_obs_dim + N_BUILDING_TYPES -> d_model``.
    Decoders map ``d_model -> 2 * local_action_dim`` (Beta params).
    When obs or action dim is 0 the corresponding linear layer is
    degenerate (bias-only or empty).

    Call :meth:`condition` to produce a :class:`Policy` bound to a
    specific building's morphology.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        extensions: ExtensionConfig | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.extensions = extensions or ExtensionConfig()

        self.encoders = nn.ModuleDict(
            {
                nt.name: nn.Linear(nt.observation_dim + N_BUILDING_TYPES, d_model)
                for nt in ALL_NODE_TYPES
            }
        )
        self.decoders = nn.ModuleDict(
            {nt.name: nn.Linear(d_model, 2 * nt.action_dim) for nt in ALL_NODE_TYPES}
        )

        if self.extensions.spectral_pe:
            self.spectral_pe_proj = nn.Linear(self.extensions.spectral_pe_k, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small init for decoders so initial policy is near-uniform Beta
        for nt in ALL_NODE_TYPES:
            if nt.action_dim > 0:
                dec = self.decoders[nt.name]
                assert isinstance(dec, nn.Linear)
                nn.init.orthogonal_(dec.weight, gain=0.01)

    def condition(self, morphology: Morphology) -> Policy:
        """Bind to a specific building's morphology."""
        nodes = morphology.nodes
        return Policy(
            morphology=morphology,
            encoders=[self.encoders[n.node_type.name] for n in nodes],
            transformer=self.transformer,
            decoders=[self.decoders[n.node_type.name] for n in nodes],
            value_head=self.value_head,
            d_model=self.d_model,
        )


# ---------------------------------------------------------------------------
# Policy (bound to a specific morphology)
# ---------------------------------------------------------------------------


@dataclass
class Policy:
    """Policy bound to a specific morphology.  Shares weights with Model.

    The token sequence equals the morphology node list: each node gets
    one encoder (obs -> d_model) and one decoder (d_model -> Beta params).
    Nodes with action_dim == 0 produce no action output.
    """

    morphology: Morphology
    encoders: list[nn.Module]  # one per node, references into Model
    transformer: nn.Module
    decoders: list[nn.Module]  # one per node, references into Model
    value_head: nn.Module
    d_model: int

    @property
    def nodes(self) -> tuple[MorphologyNode, ...]:
        return self.morphology.nodes

    @property
    def total_action_dim(self) -> int:
        return sum(n.node_type.action_dim for n in self.nodes)

    def forward(
        self,
        local_obs_tensors: list[torch.Tensor],  # per-node: (B, obs_dim_i)
        building_type_onehot: torch.Tensor,  # (B, N_BUILDING_TYPES)
    ) -> tuple[ScaledBeta, torch.Tensor]:
        """Returns ``(action_dist, value)``.

        *action_dist* is a :class:`ScaledBeta` distribution over the
        physical action space (rescaled per node type).
        """
        B = building_type_onehot.shape[0]
        device = building_type_onehot.device

        embeddings: list[torch.Tensor] = []
        for i, (node, enc) in enumerate(zip(self.nodes, self.encoders)):
            local = local_obs_tensors[i]  # (B, obs_dim)
            aug = torch.cat([local, building_type_onehot], dim=-1)
            embeddings.append(enc(aug))  # (B, d_model)

        tokens = torch.stack(embeddings, dim=1)  # (B, N_nodes, d_model)
        encoded = self.transformer(tokens)  # (B, N_nodes, d_model)

        alpha_parts: list[torch.Tensor] = []
        beta_parts: list[torch.Tensor] = []
        low_parts: list[torch.Tensor] = []
        high_parts: list[torch.Tensor] = []
        for i, (node, dec) in enumerate(zip(self.nodes, self.decoders)):
            nt = node.node_type
            if nt.action_dim == 0:
                continue
            params = dec(encoded[:, i, :])  # (B, 2 * action_dim)
            alpha_parts.append(params[:, : nt.action_dim])
            beta_parts.append(params[:, nt.action_dim :])
            space = nt.local_action_space
            low_parts.append(torch.tensor(space.low, device=device))
            high_parts.append(torch.tensor(space.high, device=device))

        assert alpha_parts, "morphology has no action-producing nodes"

        alpha = F.softplus(torch.cat(alpha_parts, dim=-1)) + 1.0
        beta_param = F.softplus(torch.cat(beta_parts, dim=-1)) + 1.0
        dist = ScaledBeta(
            Beta(alpha, beta_param), torch.cat(low_parts), torch.cat(high_parts)
        )
        value = self.value_head(encoded.mean(dim=1)).squeeze(-1)
        return dist, value


# ---------------------------------------------------------------------------
# Observation / action helpers
# ---------------------------------------------------------------------------


def encode_observation(morphology: Morphology, obs: np.ndarray) -> list[np.ndarray]:
    """Split a flat env observation into per-node local arrays.

    Returns one array per morphology node (in node order).  Nodes with
    ``observation_dim == 0`` get a zero-length array.
    """
    split = morphology.split_observation(obs)
    return [
        split.get(
            node.node_id,
            np.zeros(node.node_type.observation_dim, dtype=np.float32),
        ).astype(np.float32)
        for node in morphology.nodes
    ]


def join_action(morphology: Morphology, flat_action: np.ndarray) -> np.ndarray:
    """Scatter a flat action vector (in physical units) into the global env action.

    *flat_action* is concatenated in morphology node order (matching the
    output of ``ScaledBeta.sample()``).  This splits it back into per-node
    chunks and calls ``morphology.join_actions()`` to place them at the
    correct global indices.
    """
    per_node: dict[str, np.ndarray] = {}
    offset = 0
    for node in morphology.nodes:
        nt = node.node_type
        if nt.action_dim == 0:
            continue
        per_node[node.node_id] = flat_action[offset : offset + nt.action_dim]
        offset += nt.action_dim
    return morphology.join_actions(per_node)


# ---------------------------------------------------------------------------
# Graph helpers (spectral PE)
# ---------------------------------------------------------------------------


def adjacency_matrix(morphology: Morphology) -> np.ndarray:
    """Symmetric adjacency matrix over the morphology node sequence."""
    n = len(morphology.nodes)
    node_id_to_idx = {node.node_id: i for i, node in enumerate(morphology.nodes)}
    adj = np.zeros((n, n), dtype=np.float32)
    for edge in morphology.edges:
        si = node_id_to_idx.get(edge.source)
        ti = node_id_to_idx.get(edge.target)
        if si is not None and ti is not None:
            adj[si, ti] = 1.0
            adj[ti, si] = 1.0
    return adj


def laplacian_eigenvectors(morphology: Morphology, k: int) -> np.ndarray:
    """Smallest *k* eigenvectors of the normalized graph Laplacian.

    Returns ``(n_nodes, k)`` float32.  Zero-padded when ``n_nodes < k``.
    """
    adj = adjacency_matrix(morphology)
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    L_norm = np.eye(n, dtype=np.float32) - D_inv_sqrt @ adj @ D_inv_sqrt

    actual_k = min(k, n)
    _, eigvecs = np.linalg.eigh(L_norm)
    vecs = eigvecs[:, :actual_k].astype(np.float32)
    if actual_k < k:
        vecs = np.concatenate(
            [vecs, np.zeros((n, k - actual_k), dtype=np.float32)], axis=1
        )
    return vecs


# ---------------------------------------------------------------------------
# Tensor helpers (used by training / eval code)
# ---------------------------------------------------------------------------


def make_building_type_onehot(building_type: str, device: torch.device) -> torch.Tensor:
    """Return a ``(1, N_BUILDING_TYPES)`` one-hot tensor."""
    idx = BUILDING_TYPE_INDEX.get(building_type, 0)
    v = torch.zeros(1, N_BUILDING_TYPES, device=device)
    v[0, idx] = 1.0
    return v


def obs_to_tensors(
    local_obs: list[np.ndarray], device: torch.device
) -> list[torch.Tensor]:
    """Convert per-node numpy arrays to batched (1, dim) tensors."""
    return [
        torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
        for o in local_obs
    ]


# ---------------------------------------------------------------------------
# AmorpheusPolicy (standalone nn.Module for eval / checkpointing)
# ---------------------------------------------------------------------------


class AmorpheusPolicy(nn.Module):
    """Standalone policy wrapping a :class:`Model` for evaluation.

    Provides the ``predict(obs, deterministic)`` interface expected by
    :func:`baselines.utils.evaluation.run_episode`.  The morphology can
    be swapped via ``policy.morphology = new_morph`` for zero-shot
    transfer evaluation.
    """

    def __init__(
        self,
        morphology: Morphology,
        *,
        building_type: str = "",
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        extensions: ExtensionConfig | None = None,
        # Backward-compat alias
        embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.morphology = morphology
        self.building_type = building_type
        self.model = Model(
            d_model=embed_dim or d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            extensions=extensions,
        )

    def forward(self, obs: torch.Tensor) -> tuple[ScaledBeta, torch.Tensor]:
        """Forward pass from flat observation tensor."""
        policy = self.model.condition(self.morphology)
        B = obs.shape[0]
        device = obs.device

        local_obs: list[torch.Tensor] = []
        for node in self.morphology.nodes:
            if node.obs_indices:
                idx = torch.tensor(node.obs_indices, device=device)
                local_obs.append(obs[:, idx])
            else:
                local_obs.append(obs.new_zeros(B, node.node_type.observation_dim))

        bt = make_building_type_onehot(self.building_type, device).expand(B, -1)
        return policy.forward(local_obs, bt)

    @torch.no_grad()
    def predict(
        self,
        observation: Any,
        deterministic: bool = True,
    ) -> tuple[Any, None]:
        """Predict an env-ready action from a raw observation.

        Returns ``(action, None)`` to match the SB3-like interface
        expected by evaluation utilities.
        """
        obs_np = np.asarray(observation, dtype=np.float32)
        if obs_np.ndim == 1:
            obs_np = obs_np[np.newaxis]

        obs_t = torch.from_numpy(obs_np)
        dist, _ = self.forward(obs_t)
        action = dist.mean if deterministic else dist.sample()
        action_np = action.squeeze(0).cpu().numpy()
        env_action = join_action(self.morphology, action_np)
        return env_action, None
