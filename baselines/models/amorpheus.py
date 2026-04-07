"""Amorpheus: type-heterogeneous transformer policy for cross-domain transfer.

Implements the architecture from Section 6.2 of the Building2Building paper.
Each node type in the morphology graph has its own encoder/decoder head,
and a shared transformer aggregates information across all nodes.

This is a PyTorch ``nn.Module`` designed to be used as a custom policy
network with SB3 (via ``ActorCriticPolicy`` with custom ``features_extractor``
and ``mlp_extractor``).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from building2building.morphology import (
    ALL_NODE_TYPES,
    Morphology,
    MorphologyNode,
    NodeType,
)


class NodeEncoder(nn.Module):
    """Per-node-type encoder: local obs -> latent embedding."""

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        if input_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.learned_embedding = nn.Parameter(
                torch.randn(embed_dim) * 0.02
            )
        self._input_dim = input_dim

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        if self._input_dim > 0:
            assert x is not None
            return self.net(x)
        return self.learned_embedding.unsqueeze(0)


class NodeDecoder(nn.Module):
    """Per-node-type decoder: transformer output -> local action."""

    def __init__(self, embed_dim: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        if output_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, output_dim),
            )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.output_dim > 0:
            return self.net(h)
        return h.new_zeros(h.shape[0], 0)


class TypeEmbedding(nn.Module):
    """Learned type embedding added to each node's latent vector."""

    def __init__(self, n_types: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_types, embed_dim)

    def forward(self, type_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(type_ids)


class AmorpheusCore(nn.Module):
    """Core transformer that processes a set of node embeddings.

    Architecture:
        1. Per-type encoders map local observations to embeddings.
        2. Type embeddings are added.
        3. A small Transformer encoder aggregates across all nodes.
        4. Per-type decoders produce local actions.
        5. A value head produces V(s) from the mean of node embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        type_list = list(ALL_NODE_TYPES)
        self._type_to_idx = {nt.name: i for i, nt in enumerate(type_list)}

        self.encoders = nn.ModuleDict(
            {
                nt.name: NodeEncoder(nt.observation_dim, embed_dim)
                for nt in type_list
            }
        )
        self.decoders = nn.ModuleDict(
            {
                nt.name: NodeDecoder(embed_dim, nt.action_dim)
                for nt in type_list
            }
        )
        self.type_embed = TypeEmbedding(len(type_list), embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        morphology: Morphology,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            morphology: The environment's morphology graph.
            obs: Flat observation tensor of shape ``(batch, obs_dim)``.

        Returns:
            ``(actions, values)`` where ``actions`` has shape
            ``(batch, action_dim)`` and ``values`` has shape ``(batch, 1)``.
        """
        batch_size = obs.shape[0]
        device = obs.device
        nodes = morphology.nodes

        embeddings: list[torch.Tensor] = []
        type_ids: list[int] = []
        action_mapping: list[tuple[MorphologyNode, int]] = []

        for node in nodes:
            nt = node.node_type
            enc = self.encoders[nt.name]
            type_ids.append(self._type_to_idx[nt.name])

            if node.obs_indices:
                idx = torch.tensor(node.obs_indices, device=device)
                local_obs = obs[:, idx]
                emb = enc(local_obs)
            else:
                emb = enc().expand(batch_size, -1)

            embeddings.append(emb)
            action_mapping.append((node, len(embeddings) - 1))

        node_embs = torch.stack(embeddings, dim=1)
        type_id_tensor = torch.tensor(type_ids, device=device)
        type_emb = self.type_embed(type_id_tensor).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        node_embs = node_embs + type_emb

        h = self.transformer(node_embs)

        all_act_indices: list[int] = [
            i for n in nodes for i in n.action_indices
        ]
        action_dim = max(all_act_indices) + 1 if all_act_indices else 0
        actions = obs.new_zeros(batch_size, action_dim)

        for node, emb_idx in action_mapping:
            if not node.action_indices:
                continue
            dec = self.decoders[node.node_type.name]
            local_act = dec(h[:, emb_idx])
            idx = torch.tensor(node.action_indices, device=device)
            actions[:, idx] = local_act

        values = self.value_head(h.mean(dim=1))

        return actions, values


class AmorpheusPolicy(nn.Module):
    """Wrapper that stores morphology and provides predict() for evaluation.

    For SB3 integration, the typical approach is to use ``AmorpheusCore``
    as a custom features extractor within SB3's ``ActorCriticPolicy``.
    This class provides a simpler standalone interface.
    """

    def __init__(
        self,
        morphology: Morphology,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.morphology = morphology
        self.core = AmorpheusCore(
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers
        )
        self._action_log_std: nn.Parameter | None = None

    def _ensure_log_std(self, action_dim: int, device: torch.device) -> None:
        if self._action_log_std is None or self._action_log_std.shape[0] != action_dim:
            self._action_log_std = nn.Parameter(
                torch.zeros(action_dim, device=device)
            )

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core(self.morphology, obs)

    @torch.no_grad()
    def predict(
        self,
        observation: Any,
        deterministic: bool = True,
    ) -> tuple[Any, None]:
        import numpy as np

        obs_np = np.asarray(observation, dtype=np.float32)
        if obs_np.ndim == 1:
            obs_np = obs_np[np.newaxis]

        obs_t = torch.from_numpy(obs_np)
        actions, _ = self.forward(obs_t)
        return actions.numpy().squeeze(0), None
