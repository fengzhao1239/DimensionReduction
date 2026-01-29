"""
Factorized Spatiotemporal (Temporal Causal) Perceiver-Style Transformer Autoencoder
-------------------------------------------------------------------------------

Input  X: (B, T, N, d_in)    (point features; you can include xyz inside d_in)
Latent Z: (B, L, M, D)       (L<T temporal downsampling, M<N spatial downsampling)

Design (factorized):
  Encoder:
    1) Spatial Perceiver cross-attn per frame:  (N -> M)  => (B, T, M, D)
    2) Temporal causal downsample cross-attn:   (T -> L)  => (B, L, M, D)
  Decoder:
    1) Temporal upsample cross-attn:            (L -> T)  => (B, T, M, D)
    2) Spatial decode cross-attn per frame:     (M -> N)  => (B, T, N, d_out)

Notes:
  - Temporal causality is enforced in the encoder downsample stage via an attention mask.
  - Decoder is typically allowed to be non-causal (reconstruction uses full latent); if you
    need causal decoding, add a causal mask in TemporalUpsample similarly.

PyTorch version: uses nn.MultiheadAttention with batch_first=True.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack

from .components import FourierPositionalEmbedding
from .fsq import FSQ
from .vector_quantize import VectorQuantize


# -------------------------
# Utilities
# -------------------------

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w = nn.Linear(in_features, hidden_features)
        self.v = nn.Linear(in_features, hidden_features)
    
    def forward(self, x):
        return F.silu(self.w(x)) * self.v(x)


def _make_mlp(dim: int, hidden_mult: int = 4, dropout: float = 0.0) -> nn.Sequential:
    hidden = dim * hidden_mult
    return nn.Sequential(
        nn.Linear(dim, hidden),
        SwiGLU(hidden, hidden),
        nn.Dropout(dropout),
        nn.Linear(hidden, dim),
        nn.Dropout(dropout),
    )



class CausalConv1d(nn.Module):
    """
    Unified causal 1D convolution.

    Supports:
      - stride = 1  → causal feature extraction
      - stride > 1  → causal downsampling

    Causality guarantee:
      output[t] depends only on input[<= t * stride]
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left padding only → strict causality
        pad_left = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad_left, 0))
        return self.conv(x)


class TemporalCausalResidualBlock(nn.Module):
    """Causal TCN residual block for (B, C, T)."""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.GroupNorm(1, channels)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.conv(h)
        h = self.act(h)
        h = self.drop1(h)

        h = self.norm2(h)
        h = self.proj(h)
        h = self.drop2(h)
        return x + h


class PixelShuffle1D(nn.Module):
    """
    1D subpixel upsampling:
      (B, C*r, L) -> (B, C, L*r)
    """
    def __init__(self, upscale_factor: int):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cr, L = x.shape
        r = self.r
        if Cr % r != 0:
            raise ValueError(f"Channels {Cr} not divisible by upscale_factor {r}")
        # (B, C*r, L) -> (B, C, L*r)
        return rearrange(x, "b (c r) l -> b c (l r)", r=r)


# -------------------------
# Core Attention Blocks
# -------------------------


class CrossAttention(nn.Module):
    """
    Cross-attention block with normalization on both query and context:
      x = x + Attn(LN_q(x), LN_kv(context))
      x = x + MLP(LN(x))
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.mlp = _make_mlp(dim, hidden_mult=4, dropout=dropout)
        self.ln_out = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,                   # (B, Q, D)
        context: torch.Tensor,             # (B, K, D)
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.ln_q(x)
        kv = self.ln_kv(context)
        attn_out, _ = self.attn(query=q, key=kv, value=kv, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln_out(x))
        return x


class SelfAttention(nn.Module):
    """
    Self-attention block:
      out = x + Attn(LN(x)) then + MLP(LN(out))
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(dim)
        self.mlp = _make_mlp(dim, hidden_mult=4, dropout=dropout)
        self.mlp_ln = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.attn_ln(x)
        attn_out, _ = self.attn(query=q, key=q, value=q, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.mlp_ln(x))
        return x


# -------------------------
# Encoder Modules
# -------------------------

class SpatialPerceiverEncoder(nn.Module):
    """
    Spatial Perceiver encoder with (coord, func) input and concatenation fusion.

    Inputs:
      coord: (B, T, N, coord_dim)
      func:  (B, T, N, func_dim)

    Output:
      spatial_latent_tokens: (B, T, num_spatial_latents, latent_dim)

    Notes:
      - coord is encoded with Fourier features -> projected to latent_dim/2
      - func is projected to latent_dim/2
      - concatenate -> fused to latent_dim
      - learned latent queries cross-attend to point features per frame
    """
    def __init__(
        self,
        coord_dim: int,
        func_dim: int,
        hidden_dim: int,
        num_spatial_latents: int,
        num_heads: int,
        fourier_embed_dim: int = 64,     # must be even
        fourier_sigma: float = 1.0,
        fourier_learnable: bool = False,
        num_cross_attn_layers: int = 2,
        num_latent_self_attn_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim % 2 != 0:
            raise ValueError(
                f"hidden_dim should be even so coord/func can each map to hidden_dim/2; got {hidden_dim}"
            )
        if fourier_embed_dim % 2 != 0:
            raise ValueError(f"fourier_embed_dim must be even; got {fourier_embed_dim}")

        self.hidden_dim = hidden_dim
        self.num_spatial_latents = num_spatial_latents

        self.coord_fourier = FourierPositionalEmbedding(
            spatial_dim=coord_dim,
            embed_dim=fourier_embed_dim,
            sigma=fourier_sigma,
            learnable=fourier_learnable,
        )
        self.coord_proj = nn.Linear(fourier_embed_dim, hidden_dim // 2)

        self.func_proj = nn.Linear(func_dim, hidden_dim // 2)

        self.fuse_point_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            SwiGLU(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.spatial_latent_queries = nn.Parameter(
            torch.randn(num_spatial_latents, hidden_dim) * 0.02
        )

        self.cross_attn_blocks = nn.ModuleList([
            CrossAttention(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_cross_attn_layers)
        ])

        self.latent_self_attn_blocks = nn.ModuleList([
            SelfAttention(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_latent_self_attn_layers)
        ])

    def forward(self, coord: torch.Tensor, func: torch.Tensor) -> torch.Tensor:
        """
        coord: (B, T, N, coord_dim)
        func:  (B, T, N, func_dim)
        
        Returns:
            spatial_latent_tokens: (B, T, M, D)
        """
        if coord.shape[:3] != func.shape[:3]:
            raise ValueError(
                f"coord and func must match on (B,T,N); got {coord.shape[:3]} vs {func.shape[:3]}"
            )

        B, T, N, _ = coord.shape

        coord_feat = self.coord_fourier(coord)          # (B,T,N,fourier_embed_dim)
        coord_feat = self.coord_proj(coord_feat)        # (B,T,N,hidden_dim/2)

        func_feat = self.func_proj(func)                # (B,T,N,hidden_dim/2)

        point_embeddings = torch.cat([coord_feat, func_feat], dim=-1)  # (B,T,N,hidden_dim)
        point_embeddings = self.fuse_point_embedding(point_embeddings) # (B,T,N,hidden_dim)

        point_embeddings_bt = rearrange(point_embeddings, "b t n d -> (b t) n d")
        latents_bt = repeat(self.spatial_latent_queries, "m d -> (b t) m d", b=B, t=T)

        for blk in self.cross_attn_blocks:
            latents_bt = blk(latents_bt, context=point_embeddings_bt)

        for blk in self.latent_self_attn_blocks:
            latents_bt = blk(latents_bt)

        spatial_latent_tokens = rearrange(latents_bt, "(b t) m d -> b t m d", b=B, t=T)
        return spatial_latent_tokens


class TemporalCausalConvDownsampler(nn.Module):
    """
    Learnable causal temporal downsampling:
      (B, T, M, D) -> (B, L, M, D), with L < T.

      - requires T % L == 0 so stride s = T//L is integer
    """
    def __init__(
        self,
        hidden_dim: int,
        num_input_frames: int,
        num_latent_frames: int,
        num_pre_tcn_blocks: int = 2,
        kernel_size: int = 3,
        pre_dilations: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        T = num_input_frames
        L = num_latent_frames
        if not (1 <= L < T):
            raise ValueError(f"Require 1 <= num_latent_frames < num_input_frames, got L={L}, T={T}")
        if T % L != 0:
            raise ValueError(f"To get exact L with strided conv, require T % L == 0, got T={T}, L={L}")
        self.hidden_dim = hidden_dim
        self.num_input_frames = T
        self.num_latent_frames = L
        self.time_stride = T // L

        if pre_dilations is None:
            pre_dilations = [1, 2]

        blocks = []
        for i in range(num_pre_tcn_blocks):
            d = pre_dilations[i % len(pre_dilations)]
            blocks.append(TemporalCausalResidualBlock(hidden_dim, kernel_size=kernel_size, dilation=d, dropout=dropout))
        self.pre_tcn = nn.Sequential(*blocks)

        # learnable downsampling
        self.downsample = CausalConv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=self.time_stride,
            dilation=1,
            bias=True,
        )

        self.post_norm = nn.GroupNorm(1, hidden_dim)

    def forward(self, spatial_latent_tokens: torch.Tensor) -> torch.Tensor:
        """
        spatial_latent_tokens: (B, T, M, D)
        returns:              (B, L, M, D)
        """
        B, T, M, D = spatial_latent_tokens.shape
        x = rearrange(spatial_latent_tokens, "b t m d -> (b m) d t")
        x = self.pre_tcn(x)
        x = self.downsample(x)                    # (B*M, D, L)
        x = self.post_norm(x)

        return rearrange(x, "(b m) d l -> b l m d", b=B, m=M)


class FactorizedPerceiverEncoder(nn.Module):
    """
    Factorized spatiotemporal encoder producing temporally-downsampled latent tokens.

    Pipeline:
      1) Spatial Perceiver (per frame):
          (coord, func) -> spatial_latent_tokens
          spatial_latent_tokens: (B, num_input_frames, num_spatial_latents, latent_dim)

      2) Temporal causal downsampling (per spatial latent index):
          spatial_latent_tokens -> temporal_latent_tokens
          temporal_latent_tokens: (B, num_latent_frames, num_spatial_latents, latent_dim)

    Final output:
      temporal_latent_tokens: (B, num_latent_frames, num_spatial_latents, latent_dim)
    """
    def __init__(
        self,
        coord_dim: int,
        func_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_input_frames: int,
        num_latent_frames: int,
        num_spatial_latents: int,
        num_heads: int = 8,
        # spatial encoder knobs
        fourier_embed_dim: int = 64,
        fourier_sigma: float = 1.0,
        fourier_learnable: bool = False,
        num_spatial_cross_attn_layers: int = 2,
        num_spatial_self_attn_layers: int = 1,
        # temporal downsampler knobs
        num_pre_tcn_blocks: int = 2,
        kernel_size: int = 3,
        pre_dilations: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_input_frames = num_input_frames
        self.num_latent_frames = num_latent_frames
        self.num_spatial_latents = num_spatial_latents
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.spatial_encoder = SpatialPerceiverEncoder(
            coord_dim=coord_dim,
            func_dim=func_dim,
            hidden_dim=hidden_dim,
            num_spatial_latents=num_spatial_latents,
            num_heads=num_heads,
            fourier_embed_dim=fourier_embed_dim,
            fourier_sigma=fourier_sigma,
            fourier_learnable=fourier_learnable,
            num_cross_attn_layers=num_spatial_cross_attn_layers,
            num_latent_self_attn_layers=num_spatial_self_attn_layers,
            dropout=dropout,
        )

        self.temporal_downsampler = TemporalCausalConvDownsampler(
            hidden_dim=hidden_dim,
            num_input_frames=num_input_frames,
            num_latent_frames=num_latent_frames,
            num_pre_tcn_blocks=num_pre_tcn_blocks,
            kernel_size=kernel_size,
            pre_dilations=pre_dilations,
            dropout=dropout,
        )

        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, coord: torch.Tensor, func: torch.Tensor) -> torch.Tensor:
        """
        coord: (B, num_input_frames, num_points_per_frame, coord_dim)
        func:  (B, num_input_frames, num_points_per_frame, func_dim)

        returns:
          latent_tokens: (B, num_latent_frames, num_spatial_latents, latent_dim)
        """
        
        coord = rearrange(coord, "b t ... d -> b t (...) d")
        func = rearrange(func, "b t ... d -> b t (...) d")
        
        spatial_latent_tokens = self.spatial_encoder(coord, func)            # (B, T, M, D)
        latent_tokens = self.temporal_downsampler(spatial_latent_tokens)  # (B, L, M, D)
        out = self.latent_proj(latent_tokens)
        return out


# -------------------------
# Decoder Modules
# -------------------------

class TemporalCausalConvUpsampler(nn.Module):
    """
    Learnable causal temporal upsampling:
      (B, L, M, D) -> (B, T, M, D)

    Contract:
      - assumes T % L == 0 and uses upscale r = T//L
      - uses subpixel upsampling (learnable via a conv that expands channels) + causal refinement
    """
    def __init__(
        self,
        hidden_dim: int,
        num_output_frames: int,
        num_latent_frames: int,
        num_post_tcn_blocks: int = 2,
        kernel_size: int = 3,
        post_dilations: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        T = num_output_frames
        L = num_latent_frames
        if not (1 <= L < T):
            raise ValueError(f"Require 1 <= num_latent_frames < num_output_frames, got L={L}, T={T}")
        if T % L != 0:
            raise ValueError(f"To get exact T with upsampling, require T % L == 0, got T={T}, L={L}")
        self.hidden_dim = hidden_dim
        self.num_output_frames = T
        self.num_latent_frames = L
        self.upscale = T // L

        # expand channels to D*r, then pixel-shuffle to length T
        self.channel_expand = nn.Conv1d(hidden_dim, hidden_dim * self.upscale, kernel_size=1, bias=True)
        self.shuffle = PixelShuffle1D(upscale_factor=self.upscale)

        if post_dilations is None:
            post_dilations = [1, 2]

        blocks = []
        for i in range(num_post_tcn_blocks):
            d = post_dilations[i % len(post_dilations)]
            blocks.append(TemporalCausalResidualBlock(hidden_dim, kernel_size=kernel_size, dilation=d, dropout=dropout))
        self.post_tcn = nn.Sequential(*blocks)

        self.post_norm = nn.GroupNorm(1, hidden_dim)

    def forward(self, temporal_latent_tokens: torch.Tensor) -> torch.Tensor:
        """
        temporal_latent_tokens: (B, L, M, D)
        returns:               (B, T, M, D)
        """
        B, L, M, D = temporal_latent_tokens.shape

        z = rearrange(temporal_latent_tokens, "b l m d -> (b m) d l")
        y = self.channel_expand(z)            # (B*M, D*r, L)
        y = self.shuffle(y)                  # (B*M, D, T)
        y = self.post_tcn(y)                 # (B*M, D, T)
        y = self.post_norm(y)

        return rearrange(y, "(b m) d t -> b t m d", b=B, m=M)


class SpatialPerceiverDecoder(nn.Module):
    """
    Query-conditioned spatial decoding per frame:

      Inputs:
        per_frame_spatial_tokens: (B, num_output_frames, num_spatial_latents, hidden_dim)
        query_coord:             (B, num_output_frames, num_query_points, coord_dim)
                                  or (B, num_query_points, coord_dim)  (shared across frames)

      Output:
        reconstructed:           (B, num_output_frames, num_query_points, output_feature_dim)

      - query points are embedded (e.g., Fourier features) -> projected to latent_dim
      - query embeddings cross-attend to per-frame latent tokens (Perceiver-style decode)
    """
    def __init__(
        self,
        coord_dim: int,
        hidden_dim: int,
        output_feature_dim: int,
        num_heads: int,
        fourier_embed_dim: int = 64,
        fourier_sigma: float = 1.0,
        fourier_learnable: bool = False,
        num_spatial_cross_attn_layers: int = 2,
        num_query_self_attn_layers: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.output_feature_dim = output_feature_dim

        # Reuse your FourierPositionalEmbedding
        self.coord_fourier = FourierPositionalEmbedding(
            spatial_dim=coord_dim,
            embed_dim=fourier_embed_dim,
            sigma=fourier_sigma,
            learnable=fourier_learnable,
        )
        self.query_proj = nn.Sequential(
            nn.Linear(fourier_embed_dim, hidden_dim),
            SwiGLU(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.spatial_cross_attn_blocks = nn.ModuleList([
            CrossAttention(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_spatial_cross_attn_layers)
        ])

        self.query_self_attn_blocks = nn.ModuleList([
            SelfAttention(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_query_self_attn_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, output_feature_dim)

    def forward(self, per_frame_spatial_tokens: torch.Tensor, query_coord: torch.Tensor) -> torch.Tensor:
        """
        per_frame_spatial_tokens: (B, T, M, D)

        query_coord: (B, T, N, coord_dim)

        returns:
          reconstructed: (B, T, N, output_feature_dim)
        """
        B, T, M, D = per_frame_spatial_tokens.shape

        q = self.coord_fourier(query_coord)
        q = self.query_proj(q)

        kv = rearrange(per_frame_spatial_tokens, 'b t m d -> (b t) m d')
        q = rearrange(q, 'b t n d -> (b t) n d')

        for blk in self.spatial_cross_attn_blocks:
            q = blk(q, context=kv)

        for blk in self.query_self_attn_blocks:
            q = blk(q)

        out = self.output_proj(q)
        reconstructed = rearrange(out, '(b t) n d -> b t n d', b=B, t=T)
        return reconstructed


class FactorizedPerceiverDecoder(nn.Module):
    """
    Decoder:
      (B, num_latent_frames, num_spatial_latents, latent_dim)
        -> (B, num_output_frames, num_points_per_frame, output_feature_dim)

    Pipeline:
      1) Temporal upsample: latent-time tokens -> per-frame spatial tokens
      2) Spatial decode: per-frame spatial tokens -> per-frame points/features
    """
    def __init__(
        self,
        coord_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_output_frames: int,
        num_latent_frames: int,
        output_feature_dim: int,
        fourier_embed_dim: int = 64,
        fourier_sigma: float = 1.0,
        fourier_learnable: bool = False,
        # temporal upsampler knobs
        num_post_tcn_blocks: int = 2,
        kernel_size: int = 3,
        post_dilations: Optional[List[int]] = None,
        # spatial decoder knobs
        num_heads: int = 8,
        num_spatial_cross_attn_layers: int = 2,
        num_query_self_attn_layers: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.temporal_upsampler = TemporalCausalConvUpsampler(
            hidden_dim=hidden_dim,
            num_output_frames=num_output_frames,
            num_latent_frames=num_latent_frames,
            num_post_tcn_blocks=num_post_tcn_blocks,    
            kernel_size=kernel_size,
            post_dilations=post_dilations,
            dropout=dropout,
        )
        self.spatial_decoder = SpatialPerceiverDecoder(
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            output_feature_dim=output_feature_dim,
            num_heads=num_heads,
            fourier_embed_dim=fourier_embed_dim,
            fourier_sigma=fourier_sigma,
            fourier_learnable=fourier_learnable,
            num_spatial_cross_attn_layers=num_spatial_cross_attn_layers,
            num_query_self_attn_layers=num_query_self_attn_layers,
            dropout=dropout,
        )

    def forward(self, query_coord: torch.Tensor, latent_tokens: torch.Tensor) -> torch.Tensor:
        """
        latent_tokens: (B, num_latent_frames, num_spatial_latents, latent_dim)
        query_coord: (B, num_output_frames, num_points_per_frame, coord_dim)
        returns:
          reconstructed: (B, num_output_frames, num_points_per_frame, output_feature_dim)
        """
        b, t, *m, d = query_coord.shape
        query_coord = query_coord.reshape(b, t, -1, d)
        latent_tokens = self.latent_to_hidden(latent_tokens)
        per_frame_spatial_tokens = self.temporal_upsampler(latent_tokens)  # (B, T, M, D)
        reconstructed = self.spatial_decoder(per_frame_spatial_tokens, query_coord)             # (B, T, N, d_out)
        out = reconstructed.reshape(b, t, *m, -1)
        return out


# -------------------------
# Assembled Autoencoder
# -------------------------


class SpatiotemporalAutoencoder(nn.Module):
    """
    Full model (coord+func variant):

      encode:
        coord, func -> latent_tokens
        latent_tokens: (B, num_latent_frames, num_spatial_latents, latent_dim)

      decode:
        latent_tokens, query_coord (if provided) -> reconstructed
        reconstructed: (B, num_output_frames, num_points_per_frame, output_feature_dim)

    Set output_feature_dim=3 to reconstruct xyz only, or to func_dim / (coord_dim+func_dim) depending on your goal.
    """
    def __init__(
        self,
        # encoder input spec
        coord_dim: int,
        func_dim: int,
        num_input_frames: int,
        # token widths
        hidden_dim: int,
        latent_dim: int,
        num_latent_frames: int,
        num_spatial_latents: int,
        # decoder output spec
        output_feature_dim: int,
        # encoder knobs
        enc_num_spatial_cross_attn_layers: int = 2,
        enc_num_spatial_self_attn_layers: int = 1,
        enc_num_pre_tcn_blocks: int = 2,
        enc_kernel_size: int = 3,
        enc_pre_dilations: Optional[List[int]] = None,
        # decoder knobs
        dec_num_spatial_cross_attn_layers: int = 2,
        dec_num_query_self_attn_layers: int = 0,
        dec_num_post_tcn_blocks: int = 2,
        dec_kernel_size: int = 3,
        dec_post_dilations: Optional[List[int]] = None,
        # other knobs
        num_heads: int = 8,
        dropout: float = 0.0,
        fourier_embed_dim: int = 64,
        fourier_sigma: float = 1.0,
        fourier_learnable: bool = False,
        quantizer_cfg=None,
    ):
        super().__init__()

        self.encoder = FactorizedPerceiverEncoder(
            coord_dim=coord_dim,
            func_dim=func_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_input_frames=num_input_frames,
            num_latent_frames=num_latent_frames,
            num_spatial_latents=num_spatial_latents,
            num_heads=num_heads,
            fourier_embed_dim=fourier_embed_dim,
            fourier_sigma=fourier_sigma,
            fourier_learnable=fourier_learnable,
            num_spatial_cross_attn_layers=enc_num_spatial_cross_attn_layers,
            num_spatial_self_attn_layers=enc_num_spatial_self_attn_layers,
            num_pre_tcn_blocks=enc_num_pre_tcn_blocks,
            kernel_size=enc_kernel_size,
            pre_dilations=enc_pre_dilations,
            dropout=dropout,
        )

        self.decoder = FactorizedPerceiverDecoder(
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_output_frames=num_input_frames,
            num_latent_frames=num_latent_frames,
            output_feature_dim=output_feature_dim,
            num_heads=num_heads,
            num_query_self_attn_layers=dec_num_query_self_attn_layers,
            num_spatial_cross_attn_layers=dec_num_spatial_cross_attn_layers,
            num_post_tcn_blocks=dec_num_post_tcn_blocks,
            kernel_size=dec_kernel_size,
            post_dilations=dec_post_dilations,
            dropout=dropout,
        )

        if quantizer_cfg is not None:
            if "codebook_size" in quantizer_cfg:
                self.quantizer = VectorQuantize(**quantizer_cfg)
            elif "levels" in quantizer_cfg:
                self.quantizer = FSQ(**quantizer_cfg)
            else:
                raise ValueError("Unknown quantizer configuration")
        self.quantizer_cfg = quantizer_cfg

    def encode(self, coords, field_values):
        latents = self.encoder(coords, field_values)
        indices = None
        quant_loss = None
        
        if self.quantizer_cfg is not None:
            if isinstance(self.quantizer, VectorQuantize):
                latents, indices, quant_loss = self.quantizer(latents)
                quant_loss = quant_loss.mean()
            elif  isinstance(self.quantizer, FSQ):
                latents, indices = self.quantizer(latents)
            
        return latents, indices, quant_loss
    
    def decode(self, coords, latents):
        outputs = self.decoder(coords, latents)
        return outputs
    
    def forward(self, input_coords, input_field_values, query_coords=None, 
                return_latents=False, return_indices=False):

        latents, indices, quant_loss = self.encode(input_coords, input_field_values)
        
        if query_coords is None:
            query_coords = input_coords
        
        outputs = self.decode(query_coords, latents)
        
        # Return quantization loss if present
        total_loss = quant_loss
        
        if return_latents and return_indices:
            return outputs, latents, indices
        if return_latents:
            return outputs, latents, total_loss
        if return_indices:
            return outputs, indices
        return outputs
    




if __name__ == "__main__":
    B, T, N, d_in = 2, 16, 4096, 2
    L, M, D = 4, 64, 16
    hidden_dim = 256
    d_out = 1

    model = SpatiotemporalAutoencoder(
        coord_dim=d_in,
        func_dim=d_in,
        num_input_frames=T,
        latent_dim=D,
        num_latent_frames=L,
        num_spatial_latents=M,
        hidden_dim=hidden_dim,
        output_feature_dim=d_out,
        enc_num_spatial_cross_attn_layers=2,
        enc_num_spatial_self_attn_layers=1,
        enc_num_pre_tcn_blocks=2,
        enc_kernel_size=3,
        enc_pre_dilations=[1, 2, 4, 8],
        dec_num_spatial_cross_attn_layers=2,
        dec_num_query_self_attn_layers=0,
        dec_num_post_tcn_blocks=2,
        dec_kernel_size=3,
        dec_post_dilations=[1, 2, 4, 8],
        num_heads=8,
        dropout=0.0,
        fourier_embed_dim=64,
        fourier_sigma=1.0,
        fourier_learnable=False,
        quantizer_cfg=None,
    )

    coords = torch.randn(B, N, d_in)
    # coords = torch.randn(B, T, N, d_in)
    field_values = torch.randn(B, T, N, d_in)
    x_hat, z, loss = model(coords, field_values, return_latents=True)
    print("x_hat:", x_hat.shape)  # (B,T,N,d_out)
    print("z:", z.shape)          # (B,L,M,D)     