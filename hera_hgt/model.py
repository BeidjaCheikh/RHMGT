from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig


class MLP(nn.Module):
    def __init__(self, dims, dropout: float = 0.0, final_norm: bool = False):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.LayerNorm(dims[i + 1]), nn.GELU(), nn.Dropout(dropout)]
        if final_norm:
            layers.append(nn.LayerNorm(dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RelationAwareSelfAttention(nn.Module):
    """Multi-head graph attention with three nonredundant pairwise biases.

    score(i,j) = Q_i K_j^T / sqrt(d_h)
                 + bond_chemistry_10D(i,j)
                 + relation_type(i,j)
                 + topological_distance(i,j)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.hidden = cfg.hidden_dim
        self.heads = cfg.n_heads
        self.head_dim = cfg.hidden_dim // cfg.n_heads
        self.max_pair_dist = cfg.max_pair_dist
        self.qkv = nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 3, bias=False)
        self.out = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.bond_bias = nn.Linear(cfg.bond_dim, cfg.n_heads, bias=False)
        self.relation_bias = nn.Embedding(cfg.relation_vocab, cfg.n_heads)
        self.distance_bias = nn.Embedding(cfg.max_pair_dist + 2, cfg.n_heads)
        self.use_bond_bias = bool(cfg.use_bond_bias)
        self.use_relation_bias = bool(cfg.use_relation_bias)
        self.use_distance_bias = bool(cfg.use_distance_bias)
        self.dropout = nn.Dropout(cfg.dropout)
        self.last_attention: Optional[torch.Tensor] = None

    def forward(
        self,
        h: torch.Tensor,
        relation: torch.Tensor,
        pair_dist: torch.Tensor,
        pair_bond10: torch.Tensor,
        pair_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = h.shape
        qkv = self.qkv(h).view(B, N, 3, self.heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)  # B,H,N,d
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        # Pair topology is always preserved by pair_mask. Ablations below only
        # remove the corresponding learned bias term, so connectivity is unchanged.
        if self.use_relation_bias:
            rel = self.relation_bias(
                relation.clamp(0, self.relation_bias.num_embeddings - 1)
            ).permute(0, 3, 1, 2)
            scores = scores + rel

        if self.use_distance_bias:
            dist_idx = pair_dist.clamp(0, self.max_pair_dist + 1)
            dist = self.distance_bias(dist_idx).permute(0, 3, 1, 2)
            scores = scores + dist

        if self.use_bond_bias:
            bond = self.bond_bias(pair_bond10).permute(0, 3, 1, 2)
            scores = scores + bond

        valid_pair = pair_mask & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        scores = scores.masked_fill(~valid_pair.unsqueeze(1), torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.dropout(attn)
        self.last_attention = attn.detach()

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        out = self.out(out)
        return out * node_mask.unsqueeze(-1).to(out.dtype)


class HierarchicalTransformerLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = RelationAwareSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        inner = cfg.hidden_dim * cfg.ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(inner, cfg.hidden_dim),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h, relation, pair_dist, pair_bond10, pair_mask, node_mask):
        a = self.attn(self.norm1(h), relation, pair_dist, pair_bond10, pair_mask, node_mask)
        h = h + self.drop(a)
        h = h + self.drop(self.ffn(self.norm2(h)))
        return h * node_mask.unsqueeze(-1).to(h.dtype)


class HierarchicalGraphEncoder(nn.Module):
    """Atoms + BRICS/hERG motifs + learned global token in one graph."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.node_proj = nn.Linear(cfg.atom_dim, cfg.hidden_dim)
        self.level_embed = nn.Embedding(3, cfg.hidden_dim)
        self.motif_family_embed = nn.Embedding(cfg.motif_family_vocab, cfg.hidden_dim)
        self.size_proj = nn.Linear(1, cfg.hidden_dim)
        self.global_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        nn.init.normal_(self.global_token, mean=0.0, std=0.02)
        self.input_norm = nn.LayerNorm(cfg.hidden_dim)
        self.layers = nn.ModuleList([HierarchicalTransformerLayer(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        node_level = batch["node_level"].clamp(0, 2)
        h = self.node_proj(batch["node_x"])
        h = h + self.level_embed(node_level)

        motif_mask = (node_level == 1).unsqueeze(-1).to(h.dtype)
        family = self.motif_family_embed(
            batch["motif_family"].clamp(0, self.motif_family_embed.num_embeddings - 1)
        )
        h = h + motif_mask * (family + self.size_proj(batch["motif_size"]))

        # The global molecule is a learned token rather than an arbitrary mean of
        # raw atom features. It receives molecular information through motif->G.
        global_mask = (node_level == 2).unsqueeze(-1)
        global_init = self.global_token.expand(h.size(0), h.size(1), -1) + self.level_embed(node_level)
        h = torch.where(global_mask, global_init, h)

        h = self.input_norm(h)
        h = h * batch["node_mask"].unsqueeze(-1).to(h.dtype)
        for layer in self.layers:
            h = layer(
                h,
                batch["relation"],
                batch["pair_dist"],
                batch["pair_bond10"],
                batch["pair_mask"],
                batch["node_mask"],
            )
        h = self.final_norm(h)
        B = h.size(0)
        global_h = h[torch.arange(B, device=h.device), batch["global_index"]]
        return h, global_h


class EvidenceEncoder(nn.Module):
    """Four complementary evidence tokens: Morgan, MACCS, AtomPair, descriptors."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.morgan = MLP([cfg.morgan_dim, cfg.fp_encoder_hidden, d], cfg.dropout, final_norm=True)
        self.maccs = MLP([cfg.maccs_dim, cfg.fp_encoder_hidden, d], cfg.dropout, final_norm=True)
        self.atompair = MLP([cfg.atom_pair_fp_dim, cfg.fp_encoder_hidden, d], cfg.dropout, final_norm=True)
        self.desc = MLP([cfg.desc_dim, cfg.desc_encoder_hidden, d], cfg.dropout, final_norm=True)
        self.modality_embed = nn.Embedding(4, d)
        self.register_buffer(
            "enabled_modalities",
            torch.tensor(
                [
                    float(cfg.use_morgan),
                    float(cfg.use_maccs),
                    float(cfg.use_atompair),
                    float(cfg.use_descriptors),
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = torch.stack([
            self.morgan(batch["fp_morgan"]),
            self.maccs(batch["fp_maccs"]),
            self.atompair(batch["fp_atompair"]),
            self.desc(batch["desc"]),
        ], dim=1)
        ids = torch.arange(4, device=tokens.device).unsqueeze(0)
        tokens = tokens + self.modality_embed(ids)

        # Zero AFTER adding the modality embedding: a removed modality contributes
        # no molecule-specific signal and no learned constant token to the fusion.
        mask = self.enabled_modalities.to(device=tokens.device, dtype=tokens.dtype)
        return tokens * mask.view(1, 4, 1)


class CrossModalFusion(nn.Module):
    """One cross-attention: global graph token queries complementary evidence."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            cfg.hidden_dim,
            cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        inner = cfg.hidden_dim * cfg.ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(inner, cfg.hidden_dim),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, global_h: torch.Tensor, evidence_tokens: torch.Tensor):
        q = global_h.unsqueeze(1)
        context, weights = self.attn(
            q,
            evidence_tokens,
            evidence_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        z = self.norm1(global_h + self.drop(context.squeeze(1)))
        z = self.norm2(z + self.drop(self.ffn(z)))
        return z, weights.squeeze(2)  # B,H,4


class ConcatFusion(nn.Module):
    """Late concatenation baseline preserving all five molecular representations.

    Input: h_G plus the four already-projected evidence tokens
    [Morgan, MACCS, AtomPair, Descriptors]. No modality is removed; only the
    cross-attention operator is replaced by direct late concatenation + MLP.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.integration = MLP(
            [d * 5, d * 2, d],
            dropout=cfg.dropout,
            final_norm=True,
        )

    def forward(self, global_h: torch.Tensor, evidence_tokens: torch.Tensor):
        B = global_h.size(0)
        flat = evidence_tokens.reshape(B, -1)
        fused = self.integration(torch.cat([global_h, flat], dim=-1))
        # Shape Bx1x0 keeps the evaluation pipeline backward compatible without
        # fabricating attention values for a model that has no attention fusion.
        no_cross_attention = global_h.new_empty((B, 1, 0))
        no_fusion_weights = global_h.new_empty((B, 0))
        return fused, no_cross_attention, no_fusion_weights


class FHGNAdaptiveFusion(nn.Module):
    """FH-GNN-inspired adaptive graph/evidence fusion.

    The four HERA-HGT evidence tokens are first summarized into one evidence
    vector h_E. Then molecule-specific two-way weights balance h_G and h_E.
    This preserves HERA-HGT's inputs while adapting the graph-vs-fingerprint
    fusion principle described by FH-GNN.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.evidence_summary = MLP(
            [d * 4, d * 2, d],
            dropout=cfg.dropout,
            final_norm=True,
        )
        # W_G and W_E in the weighted fusion equation.
        self.graph_proj = nn.Linear(d, d, bias=False)
        self.evidence_proj = nn.Linear(d, d, bias=False)
        self.gate = MLP([d * 2, d, 2], dropout=cfg.dropout)
        self.bias = nn.Parameter(torch.zeros(d))
        self.out_norm = nn.LayerNorm(d)

    def summarize_evidence(self, evidence_tokens: torch.Tensor) -> torch.Tensor:
        B = evidence_tokens.size(0)
        return self.evidence_summary(evidence_tokens.reshape(B, -1))

    def forward(self, global_h: torch.Tensor, evidence_tokens: torch.Tensor):
        B = global_h.size(0)
        h_e = self.summarize_evidence(evidence_tokens)
        gates = torch.softmax(self.gate(torch.cat([global_h, h_e], dim=-1)), dim=-1)
        h_g_proj = self.graph_proj(global_h)
        h_e_proj = self.evidence_proj(h_e)
        fused = (
            gates[:, 0:1] * h_g_proj
            + gates[:, 1:2] * h_e_proj
            + self.bias
        )
        fused = self.out_norm(fused)
        no_cross_attention = global_h.new_empty((B, 1, 0))
        return fused, no_cross_attention, gates


class HERAHGT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.graph_encoder = HierarchicalGraphEncoder(cfg)
        self.evidence_encoder = EvidenceEncoder(cfg)

        if cfg.fusion_type == "cross_attention":
            self.fusion = CrossModalFusion(cfg)
            self.fusion_weight_names = []
        elif cfg.fusion_type == "concat":
            self.fusion = ConcatFusion(cfg)
            self.fusion_weight_names = []
        elif cfg.fusion_type == "fhgnn_adaptive":
            self.fusion = FHGNAdaptiveFusion(cfg)
            self.fusion_weight_names = ["graph", "evidence"]
        else:  # guarded by cfg.validate(); defensive for custom callers
            raise ValueError(f"Unknown fusion_type={cfg.fusion_type!r}")

        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.classifier_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.classifier_hidden, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mode = self.cfg.prediction_mode
        device = batch["node_x"].device
        B = batch["node_x"].size(0)

        empty_repr = torch.empty((B, 0), device=device)
        empty_nodes = torch.empty((B, 0, self.cfg.hidden_dim), device=device)
        empty_cross = torch.empty((B, 1, 0), device=device)
        empty_fusion = torch.empty((B, 0), device=device)
        empty_attn = torch.empty(0, device=device)

        if mode == "graph_only":
            node_h, global_h = self.graph_encoder(batch)
            evidence = torch.empty((B, 0, self.cfg.hidden_dim), device=device)
            fused = global_h
            cross_weights = empty_cross
            fusion_weights = empty_fusion
            last_hier_attn = self.graph_encoder.layers[-1].attn.last_attention

        elif mode == "evidence_only":
            # For a fair single-view baseline, reuse the exact evidence-summary
            # subnetwork used by the selected FH-GNN-inspired full model.
            evidence = self.evidence_encoder(batch)
            global_h = empty_repr
            node_h = empty_nodes
            fused = self.fusion.summarize_evidence(evidence)
            cross_weights = empty_cross
            fusion_weights = empty_fusion
            last_hier_attn = None

        else:
            node_h, global_h = self.graph_encoder(batch)
            evidence = self.evidence_encoder(batch)

            if self.cfg.fusion_type == "cross_attention":
                fused, cross_weights = self.fusion(global_h, evidence)
                fusion_weights = empty_fusion
            else:
                fused, cross_weights, fusion_weights = self.fusion(global_h, evidence)

            last_hier_attn = self.graph_encoder.layers[-1].attn.last_attention

        logits = self.classifier(fused).squeeze(-1)
        return {
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "global_repr": global_h,
            "fused_repr": fused,
            "evidence_tokens": evidence,
            "cross_attention": cross_weights,
            "fusion_weights": fusion_weights,
            "hierarchical_attention": (
                last_hier_attn if last_hier_attn is not None else empty_attn
            ),
            "node_embeddings": node_h,
        }
