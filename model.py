"""ACT model — a CVAE whose decoder is the policy (Zhao et al., 2304.13705).

Three transformer pieces (the naming trips people up):
  - vae_encoder : BERT-style encoder, TRAINING ONLY. Sees the ground-truth
                  action chunk + state and squeezes the demo "style" into z.
                  Discarded at test time (z := 0, the prior mean).
  - encoder     : the policy's perception encoder. Fuses [z, state, image tokens]
                  into memory. Used at train AND test.
  - decoder     : cross-attends k learned query slots to that memory and emits
                  the k-step action chunk.

Positional embeddings follow DETR/ACT: pos is added to query/key at *every*
attention layer (not just the input), so we use custom layers, not nn.Transformer.
"""
import math
from collections import deque

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from config import ACTConfig


# =============================================================================
# Normalization (mean/std), owned by the model so it travels with the checkpoint
# =============================================================================
def _to_tensor(x) -> Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


class Normalize(nn.Module):
    """(x - mean) / std for the given keys, using dataset stats as buffers."""

    def __init__(self, stats: dict, keys: list[str]):
        super().__init__()
        self._keys = list(keys)
        for key in self._keys:
            safe = key.replace(".", "_")
            self.register_buffer(f"{safe}__mean", _to_tensor(stats[key]["mean"]))
            self.register_buffer(f"{safe}__std", _to_tensor(stats[key]["std"]).clamp_min(1e-8))

    def _get(self, key):
        safe = key.replace(".", "_")
        return getattr(self, f"{safe}__mean"), getattr(self, f"{safe}__std")

    def forward(self, batch: dict) -> dict:
        out = dict(batch)
        for key in self._keys:
            if key not in out:
                continue
            mean, std = self._get(key)
            # action (B,k,D) broadcasts over the chunk; state (B,D); image (B,3,H,W)
            out[key] = (out[key] - mean) / std
        return out


class Unnormalize(nn.Module):
    """x * std + mean — maps predicted actions back to control units."""

    def __init__(self, stats: dict, key: str = "action"):
        super().__init__()
        self.register_buffer("mean", _to_tensor(stats[key]["mean"]))
        self.register_buffer("std", _to_tensor(stats[key]["std"]).clamp_min(1e-8))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.std + self.mean


# =============================================================================
# Positional embeddings
# =============================================================================
def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal position embedding (Attention Is All You Need)."""
    def angle_vec(pos):
        return [pos / np.power(10000, 2 * (j // 2) / dimension) for j in range(dimension)]
    table = np.array([angle_vec(p) for p in range(num_positions)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.from_numpy(table).float()


class SinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal embedding for the (H,W) image feature grid (DETR style)."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        # x is the CNN feature map (B,C,H,W). A transformer is permutation-
        # invariant, so we must hand it the (row,col) of every patch.
        # Build a coordinate grid the DETR way: start from a grid of ones and
        # cumsum it. cumsum down rows -> 1,2,...,H (the y-coordinate of each
        # patch); cumsum along cols -> 1,2,...,W (the x-coordinate). "not_mask"
        # is a DETR vestige (no real padding here, so every cell is 1).
        not_mask = torch.ones_like(x[0, :1])  # (1,H,W)
        y_range = not_mask.cumsum(1, dtype=torch.float32)  # row index per patch
        x_range = not_mask.cumsum(2, dtype=torch.float32)  # col index per patch
        # Normalize each axis by its max ([:, -1:, :] = last row H; [:, :, -1:]
        # = last col W) so coords land in (0, 2pi] regardless of feature-map size.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi
        # Standard sinusoidal frequency bands (same as the 1D embedding).
        inv_freq = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )
        x_range = x_range.unsqueeze(-1) / inv_freq
        y_range = y_range.unsqueeze(-1) / inv_freq
        # sin on even channels, cos on odd, for each axis; concat the y-half and
        # x-half -> a (1,C,H,W) embedding, one C-vector per spatial location.
        pe_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pe_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        return torch.cat((pe_y, pe_x), dim=3).permute(0, 3, 1, 2)  # (1,C,H,W)


def _activation(name: str):
    return {"relu": F.relu, "gelu": F.gelu, "glu": F.glu}[name]


# =============================================================================
# Transformer encoder / decoder (pos added at every attention; seq-first layout)
# =============================================================================
class EncoderLayer(nn.Module):
    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.dim_model, cfg.n_heads, dropout=cfg.dropout)
        self.linear1 = nn.Linear(cfg.dim_model, cfg.dim_feedforward)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.dim_model)
        self.norm1 = nn.LayerNorm(cfg.dim_model)
        self.norm2 = nn.LayerNorm(cfg.dim_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.activation = _activation(cfg.feedforward_activation)
        self.pre_norm = cfg.pre_norm

    def forward(self, x, pos_embed=None, key_padding_mask=None):
        """One transformer encoder block. Shape is preserved end-to-end.

        Layout is SEQUENCE-FIRST (what nn.MultiheadAttention expects):
            S = sequence length (perception enc: 902; vae enc: 102)
            B = batch    D = dim_model = 512    F = dim_feedforward = 3200
        Inputs:
            x                (S, B, D)  token features.
            pos_embed        (S, 1, D)  positional codes, broadcast over batch;
                                        added to q,k only (None -> no position).
            key_padding_mask (B, S)     True where a key is padding (ignored).
        Returns:
            x  (S, B, D)  same shape, contextualized by self-attention + MLP.
        """
        # Two sublayers: (1) self-attention, (2) feed-forward MLP. Each is a
        # residual block x = x + Sublayer(x). norm1 belongs to the attention
        # sublayer, norm2 to the MLP -- they are SEPARATE LayerNorms with their
        # own learned scale/shift. pre_norm decides WHERE the norm sits:
        #   pre-norm : x = x + Sublayer(LayerNorm(x))  -- norm before sublayer,
        #              clean residual highway -> stable, no LR warmup needed.
        #   post-norm: x = LayerNorm(x + Sublayer(x))  -- norm after the add
        #              (the original Transformer; ACT default).
        skip = x                          # (S,B,D) residual branch
        if self.pre_norm:
            x = self.norm1(x)             # (S,B,D) pre-norm: normalize attn input
        # Self-attention: query and key carry POSITION (q=k=x+pos) so attention
        # weights know token order; value stays position-free (=x) so the mixed
        # payload is pure content. pos is re-added every layer (DETR), not once.
        q = k = x if pos_embed is None else x + pos_embed  # (S,B,D)
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]  # (S,B,D)
        x = skip + self.dropout1(x)       # (S,B,D) residual add for the attention sublayer
        if self.pre_norm:
            skip = x                      # (S,B,D) new clean residual for the MLP
            x = self.norm2(x)             # (S,B,D) pre-norm: normalize MLP input
        else:
            x = self.norm1(x)             # (S,B,D) post-norm: finalize the attention sublayer...
            skip = x                      # ...and that normed output is the residual
        # MLP: D -> F -> D (the 3200-d hidden expansion lives only inside here).
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))  # (S,B,D)->(S,B,F)->(S,B,D)
        x = skip + self.dropout2(x)       # (S,B,D) residual add for the MLP sublayer
        if not self.pre_norm:
            x = self.norm2(x)             # (S,B,D) post-norm: finalize the MLP sublayer
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: ACTConfig, is_vae: bool = False):
        super().__init__()
        n = cfg.n_vae_encoder_layers if is_vae else cfg.n_encoder_layers
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(n)])
        self.norm = nn.LayerNorm(cfg.dim_model) if cfg.pre_norm else nn.Identity()

    def forward(self, x, pos_embed=None, key_padding_mask=None):
        """Stack of N encoder blocks. In/out both (S, B, D); shape unchanged.
        pos_embed (S,1,D) is re-fed to EVERY layer. Final self.norm is a real
        LayerNorm only in pre-norm mode, else Identity (post-norm blocks already
        normed their own output)."""
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)  # (S,B,D)
        return self.norm(x)               # (S,B,D)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.dim_model, cfg.n_heads, dropout=cfg.dropout)
        self.cross_attn = nn.MultiheadAttention(cfg.dim_model, cfg.n_heads, dropout=cfg.dropout)
        self.linear1 = nn.Linear(cfg.dim_model, cfg.dim_feedforward)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.dim_model)
        self.norm1 = nn.LayerNorm(cfg.dim_model)
        self.norm2 = nn.LayerNorm(cfg.dim_model)
        self.norm3 = nn.LayerNorm(cfg.dim_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.dropout3 = nn.Dropout(cfg.dropout)
        self.activation = _activation(cfg.feedforward_activation)
        self.pre_norm = cfg.pre_norm

    @staticmethod
    def _add(t, pos):
        return t if pos is None else t + pos

    def forward(self, x, mem, decoder_pos_embed=None, encoder_pos_embed=None):
        """One transformer decoder block. Query length stays k throughout.

        Sequence-first layout:
            k    = chunk_size = 100  (number of action query slots)
            Senc = encoder seq = 902 (the memory length)
            B = batch    D = dim_model = 512    F = dim_feedforward = 3200
        Inputs:
            x                 (k, B, D)    the query-slot features (start as 0).
            mem               (Senc, B, D) encoder output (fused observation).
            decoder_pos_embed (k, 1, D)    learned position of each query slot.
            encoder_pos_embed (Senc, 1, D) position of each memory token.
        Returns:
            x  (k, B, D)  query slots after attending to themselves + the memory.

        Three sublayers (norm1/2/3 are three separate LayerNorms):
          1) self-attn  among the k action-query slots
          2) cross-attn from the queries INTO the encoder memory
          3) feed-forward MLP
        `mem` = the encoder output: the fused [latent z, state, image patch]
        tokens. The decoder never re-reads raw inputs; it attends into `mem`.
        """
        skip = x                          # (k,B,D)
        if self.pre_norm:
            x = self.norm1(x)             # (k,B,D)
        # (1) self-attn over the query slots; position = learned decoder_pos_embed.
        q = k = self._add(x, decoder_pos_embed)  # (k,B,D)
        x = self.self_attn(q, k, value=x)[0]     # (k,B,D)
        x = skip + self.dropout1(x)       # (k,B,D)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)             # (k,B,D)
        else:
            x = self.norm1(x)             # (k,B,D)
            skip = x
        # (2) cross-attn. query and key live in DIFFERENT spaces, so each needs
        # its own positional code:
        #   decoder_pos_embed -> which of the k action slots a query is.
        #   encoder_pos_embed -> where each memory token sits (the SAME enc_pos
        #                        that was added when memory was built).
        # value=mem stays position-free (pure content), like in self-attention.
        # query is (k,B,D), key/value are (Senc,B,D); attention output is (k,B,D).
        x = self.cross_attn(
            query=self._add(x, decoder_pos_embed),     # (k,B,D)
            key=self._add(mem, encoder_pos_embed),     # (Senc,B,D)
            value=mem,                                 # (Senc,B,D)
        )[0]                                           # (k,B,D)
        x = skip + self.dropout2(x)       # (k,B,D)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)             # (k,B,D)
        else:
            x = self.norm2(x)             # (k,B,D)
            skip = x
        # (3) feed-forward MLP: D -> F -> D.
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))  # (k,B,D)->(k,B,F)->(k,B,D)
        x = skip + self.dropout3(x)       # (k,B,D)
        if not self.pre_norm:
            x = self.norm3(x)             # (k,B,D)
        return x


class Decoder(nn.Module):
    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_decoder_layers)])
        self.norm = nn.LayerNorm(cfg.dim_model)

    def forward(self, x, mem, decoder_pos_embed=None, encoder_pos_embed=None):
        """Stack of N decoder blocks (N=1 here, matching the ACT paper).
        x stays (k, B, D) throughout; mem is (Senc, B, D). Always applies a
        final LayerNorm before the action head reads it."""
        for layer in self.layers:
            x = layer(x, mem, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)  # (k,B,D)
        return self.norm(x)               # (k,B,D)


# =============================================================================
# ACT network
# =============================================================================
class ACT(nn.Module):
    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.cfg = cfg

        # --- CVAE encoder (training only) ---------------------------------
        if cfg.use_vae:
            self.vae_encoder = Encoder(cfg, is_vae=True)
            self.vae_cls_embed = nn.Embedding(1, cfg.dim_model)
            self.vae_state_proj = nn.Linear(cfg.state_dim, cfg.dim_model)
            self.vae_action_proj = nn.Linear(cfg.action_dim, cfg.dim_model)
            self.vae_latent_proj = nn.Linear(cfg.dim_model, cfg.latent_dim * 2)  # -> mu, log_var
            # fixed sinusoidal pos for [cls, state, *actions] = chunk_size + 2 tokens
            self.register_buffer(
                "vae_pos_enc",
                create_sinusoidal_pos_embedding(cfg.chunk_size + 2, cfg.dim_model).unsqueeze(0),
            )

        # --- vision backbone (shared across cameras) ----------------------
        # FrozenBatchNorm: BN running stats are unreliable at batch_size ~8, so
        # we freeze them (DETR/ACT convention) and let other layers fine-tune.
        backbone = getattr(torchvision.models, cfg.vision_backbone)(
            weights=cfg.pretrained_backbone_weights, norm_layer=FrozenBatchNorm2d
        )
        # RGBD: expand conv1 from 3->4 input channels. Keep pretrained RGB
        # weights on channels [0..2]; init channel 3 (depth) as the mean of the
        # RGB filters so pretrained edge/blob detectors respond to depth at t=0
        # instead of starting from noise (standard depth-estimation warm-start).
        if cfg.use_depth:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels=4,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            backbone.conv1 = new_conv
        self.backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})
        backbone_out_ch = backbone.fc.in_features  # 512 for resnet18

        # --- perception encoder -------------------------------------------
        self.encoder = Encoder(cfg)
        self.encoder_latent_proj = nn.Linear(cfg.latent_dim, cfg.dim_model)
        self.encoder_state_proj = nn.Linear(cfg.state_dim, cfg.dim_model)
        self.encoder_img_proj = nn.Conv2d(backbone_out_ch, cfg.dim_model, kernel_size=1)
        self.encoder_1d_pos = nn.Embedding(2, cfg.dim_model)  # [latent, state] tokens
        self.encoder_cam_pos = SinusoidalPositionEmbedding2d(cfg.dim_model // 2)

        # --- action decoder -----------------------------------------------
        self.decoder = Decoder(cfg)
        self.decoder_pos_embed = nn.Embedding(cfg.chunk_size, cfg.dim_model)  # k learned queries
        self.action_head = nn.Linear(cfg.dim_model, cfg.action_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier on transformer params (DETR convention); leave backbone pretrained.
        for name, p in self.named_parameters():
            if "backbone" not in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode_latent(self, state, actions, action_is_pad):
        """CVAE encoder (training only): squeeze a demo into the latent z.

        Symbol legend (concrete values for this project):
            B  = batch size            (8 in training)
            k  = chunk_size            = 100  action steps per query
            A  = action_dim            = 8    (7 arm joint targets + gripper)
            S  = state_dim             = 8    (7 arm joints + gripper finger)
            D  = dim_model             = 512  transformer width
            L  = latent_dim            = 32   size of z

        Inputs:
            state         (B, S)    current robot joint+gripper positions.
            actions       (B, k, A) the GROUND-TRUTH action chunk from the demo.
            action_is_pad (B, k)    True where a step is padding (episode ran out).
        Returns:
            mu      (B, L)  mean of the posterior q(z | state, actions).
            log_var (B, L)  log-variance of that posterior.
        Meaning: the [CLS] token summarizes "what style of motion is this demo",
        compressed into a 32-d Gaussian. Discarded at test time (z := 0).
        """
        bs = state.shape[0]
        # Build the token sequence [CLS, state, action_0..action_{k-1}].
        cls = einops.repeat(self.vae_cls_embed.weight, "1 d -> b 1 d", b=bs)  # (B,1,D)
        state_tok = self.vae_state_proj(state).unsqueeze(1)         # (B,1,D)
        action_tok = self.vae_action_proj(actions)                 # (B,k,D)
        x = torch.cat([cls, state_tok, action_tok], dim=1)         # (B,k+2,D)  k+2=102 tokens
        pos = self.vae_pos_enc.clone().detach()                    # (1,k+2,D)  fixed sinusoidal
        # Do not let padded action steps influence the [CLS] summary.
        cls_state_pad = torch.zeros(bs, 2, dtype=torch.bool, device=state.device)  # (B,2) CLS+state never pad
        key_pad = torch.cat([cls_state_pad, action_is_pad], dim=1)  # (B,k+2)
        # nn.MultiheadAttention wants seq-first: permute (B,k+2,D) -> (k+2,B,D).
        out = self.vae_encoder(
            x.permute(1, 0, 2), pos_embed=pos.permute(1, 0, 2), key_padding_mask=key_pad
        )                                                          # (k+2,B,D)
        cls_out = out[0]                                           # (B,D)  the [CLS] token row
        params = self.vae_latent_proj(cls_out)                     # (B,2L)  -> split into mu|log_var
        return params[:, : self.cfg.latent_dim], params[:, self.cfg.latent_dim :]  # (B,L),(B,L)

    def forward(self, batch: dict):
        """Full policy forward: observation (+demo at train) -> action chunk.

        Symbol legend (concrete values for this project):
            B  = batch size       (8 train / 1 rollout)   k  = chunk_size = 100
            A  = action_dim = 8   S = state_dim = 8        D  = dim_model = 512
            L  = latent_dim = 32  C = 3 cameras (front/diag/wrist)
            h,w = feature-map grid; resnet18 layer4 on 480x640 -> 15 x 20 = 300
            hw = 300 patches/cam  ->  Senc = 2 + C*hw = 2 + 3*300 = 902 tokens

        Inputs (dict `batch`):
            observation.state            (B, S)         current joint+gripper pos.
            observation.images.<cam>     (B, 3, 480, 640) RGB in [0,1], per camera.
            action      (B, k, A)  } TRAIN ONLY: ground-truth chunk for the CVAE
            action_is_pad (B, k)   } encoder + the L1 loss target.
        Returns:
            actions  (B, k, A)  predicted action chunk (NORMALIZED units here;
                                the policy wrapper un-normalizes to control units).
            mu, log_var  (B, L) each  -- posterior params for the KL loss; both
                                None at eval / when VAE is off.
        """
        cfg = self.cfg
        state = batch["observation.state"]                          # (B,S)
        bs = state.shape[0]

        # 1) latent z --------------------------------------------------------
        if cfg.use_vae and self.training and "action" in batch:
            mu, log_var = self._encode_latent(state, batch["action"], batch["action_is_pad"])  # (B,L),(B,L)
            # Reparameterization trick: z = mu + sigma * eps,  eps ~ N(0, I).
            # The net outputs log_var = log(sigma^2) (not the variance itself):
            # log_var ranges over all reals, and exp() maps it to (0, inf) so the
            # std is always positive and numerically stable. Hence
            #   sigma = sqrt(exp(log_var)) = exp(0.5 * log_var).
            # So (0.5*log_var).exp() == sigma and randn_like(mu) == eps.
            # Variance must be strictly positive, but a linear layer outputs any real number. If you regressed 
            # σ directly you'd need to clamp/softplus it and risk zero or negative values. log_var ranges over all 
            # of ℝ and exp() maps it to (0, ∞), guaranteeing positivity for free and staying numerically stable 
            # across many orders of magnitude. It also makes the KL term clean: 
            # KL = -½ Σ(1 + log_var − μ² − exp(log_var)) (your compute_loss), which is expressed directly in log_var.

            z = mu + (0.5 * log_var).exp() * torch.randn_like(mu)   # (B,L)
        else:
            # Eval / inference: use the prior mean z = 0 (deterministic).
            mu = log_var = None
            z = torch.zeros(bs, cfg.latent_dim, device=state.device, dtype=state.dtype)  # (B,L)

        # 2) encoder tokens: [latent, state, *image_patches]  (each row is (B,D)) -
        tokens = [self.encoder_latent_proj(z), self.encoder_state_proj(state)]  # 2 x (B,D)
        pos = list(self.encoder_1d_pos.weight.unsqueeze(1))        # 2 x (1,D)  learned 1D pos
        for cam in cfg.cameras:
            rgb = batch[f"observation.images.{cam}"]               # (B,3,H,W) normalized
            if cfg.use_depth:
                # Decoded depth video has 3 identical channels (uint8 tile).
                # Take one -> (B,1,H,W), then concat with RGB -> (B,4,H,W).
                dep = batch[f"observation.depths.{cam}"][:, :1]    # (B,1,H,W) normalized
                cam_in = torch.cat([rgb, dep], dim=1)              # (B,4,H,W)
            else:
                cam_in = rgb                                       # (B,3,H,W)
            feat = self.backbone(cam_in)["feature_map"]            # (B,512,h,w)
            cam_pos = self.encoder_cam_pos(feat).to(dtype=feat.dtype)  # (1,D,h,w) 2D sinusoidal
            feat = self.encoder_img_proj(feat)                     # (B,D,h,w)  1x1 conv 512->D
            feat = einops.rearrange(feat, "b c h w -> (h w) b c")  # (hw,B,D)   flatten grid to tokens
            cam_pos = einops.rearrange(cam_pos, "b c h w -> (h w) b c")  # (hw,1,D)
            tokens.extend(list(feat))                              # +hw x (B,D) per camera
            pos.extend(list(cam_pos))
        enc_in = torch.stack(tokens, dim=0)                        # (Senc,B,D)  Senc=902
        enc_pos = torch.stack(pos, dim=0)                          # (Senc,1,D)

        # 3) encode -> memory, then decode k actions -------------------------
        memory = self.encoder(enc_in, pos_embed=enc_pos)           # (Senc,B,D)  fused context
        dec_in = torch.zeros(cfg.chunk_size, bs, cfg.dim_model,
                             dtype=enc_pos.dtype, device=enc_pos.device)  # (k,B,D) zero queries
        dec_out = self.decoder(
            dec_in, memory,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),  # (k,1,D) learned slot pos
            encoder_pos_embed=enc_pos,                             # (Senc,1,D) memory pos
        )                                                          # (k,B,D)
        actions = self.action_head(dec_out.transpose(0, 1))        # (k,B,D)->(B,k,D)->(B,k,A)
        return actions, mu, log_var



# =============================================================================
# Temporal ensembler (inference only) — online weighted average, ACT Algorithm 2
# =============================================================================
class TemporalEnsembler:
    """Weighted-average overlapping chunk predictions for the same timestep.

    w_i = exp(-coeff * i), i=0 == oldest prediction (so coeff>0 weighs OLD actions
    more — the ACT default 0.01 does exactly this). Query the policy every step,
    feed the fresh (B,k,A) chunk in, and one ensembled action is popped per call.
    Online form (no history cache); mirrors lerobot's ACTTemporalEnsembler.
    """

    def __init__(self, coeff: float, chunk_size: int):
        self.chunk_size = chunk_size
        self.weights = torch.exp(-coeff * torch.arange(chunk_size))
        self.weights_cumsum = torch.cumsum(self.weights, dim=0)
        self.reset()

    def reset(self):
        self.ensembled = None          # (B, remaining, A)
        self.count = None              # (remaining, 1) #predictions averaged per slot

    @torch.no_grad()
    def update(self, actions: Tensor) -> Tensor:
        """actions: (B, k, A) fresh prediction. Returns (B, A) ensembled action."""
        self.weights = self.weights.to(actions.device)
        self.weights_cumsum = self.weights_cumsum.to(actions.device)
        # First chunk ever: nothing to average against, just store it. count[j]
        # = how many predictions have contributed to pending slot j (all 1 now).
        if self.ensembled is None:
            self.ensembled = actions.clone()                       # (B,k,A) running averages or (B, remaining, A)
            self.count = torch.ones((self.chunk_size, 1), dtype=torch.long, device=actions.device) # (remaining, 1)
        else:
            # Online running weighted average, no history cached. If a slot holds
            # the average of its first c predictions, then
            #   average = weighted_sum / weights_cumsum[c-1].
            # Recover the sum, fold in the new prediction (its weight is the
            # c-th, 0-indexed), then renormalize by the new cumulative weight:
            self.ensembled *= self.weights_cumsum[self.count - 1]  # avg -> weighted SUM (undo the previous normalization)
            self.ensembled += actions[:, :-1] * self.weights[self.count]  # + new term (1st k-1 align) adds new prediction, weighted by its rank (count-th, 0-idx)
            self.ensembled /= self.weights_cumsum[self.count]      # -> updated average re-normalize: divide by new cumulative weight -> avg again
            self.count = torch.clamp(self.count + 1, max=self.chunk_size)  # one more contributor each or one more prediction now contributes to each slot
            # The last action of the new chunk targets a timestep no prior chunk
            # reached, so it has no running average yet -> append it with count 1.
            self.ensembled = torch.cat([self.ensembled, actions[:, -1:]], dim=1) # the farthest-future slot is brand new
            self.count = torch.cat([self.count, torch.ones_like(self.count[-1:])]) # # ...so it starts with count=1
        # Slot 0 = current timestep, now fully averaged -> emit it and shift the
        # whole buffer one step into the future.
        action = self.ensembled[:, 0] # slot for the CURRENT timestep is fully averaged -> emit it
        self.ensembled = self.ensembled[:, 1:] # drop it; everything shifts one step into the future
        self.count = self.count[1:]
        return action


# =============================================================================
# Policy wrapper — normalization + loss (train) + action selection (rollout)
# =============================================================================
class ACTPolicy(nn.Module):
    def __init__(self, cfg: ACTConfig, stats: dict):
        super().__init__()
        self.cfg = cfg
        self.model = ACT(cfg)
        image_keys = [f"observation.images.{c}" for c in cfg.cameras]
        depth_keys = [f"observation.depths.{c}" for c in cfg.cameras] if cfg.use_depth else []
        self.normalize = Normalize(stats, ["observation.state", "action", *image_keys, *depth_keys])
        self.normalize_inputs = Normalize(stats, ["observation.state", *image_keys, *depth_keys])
        self.unnormalize_action = Unnormalize(stats, "action")
        self._ensembler = None

    # --- training -------------------------------------------------------------
    def compute_loss(self, batch: dict) -> tuple[Tensor, dict]:
        """One training step loss.

        Input `batch` (raw, un-normalized) holds:
            observation.state         (B, S=8)
            observation.images.<cam>  (B, 3, 480, 640)  x3
            action                    (B, k=100, A=8)   ground-truth target chunk
            action_is_pad             (B, k)            True on padding steps
        Returns:
            loss  scalar tensor  -- L1 + kl_weight*KL, what backprop runs on.
            out   dict of floats {l1_loss, kld_loss, loss} for logging.
        """
        batch = self.normalize(batch)                                   # in-place style: standardizes state/action/images
        actions_hat, mu, log_var = self.model(batch)                    # (B,k,A), (B,L), (B,L)

        # Per-element L1 between predicted and true normalized actions.
        l1 = F.l1_loss(batch["action"], actions_hat, reduction="none")  # (B,k,A)
        mask = ~batch["action_is_pad"]                                  # (B,k)  keep only real steps
        # Mask out padded steps, then average over (real steps * action dims).
        l1 = (l1 * mask.unsqueeze(-1)).sum() / (mask.sum() * actions_hat.shape[-1]).clamp_min(1)  # scalar

        loss = l1
        out = {"l1_loss": l1.item()}
        if self.cfg.use_vae and mu is not None:
            # KL( N(mu, sigma^2) || N(0, I) ): closed form, summed over the L
            # latent dims (-> (B,)), then mean over the batch (-> scalar). Pulls
            # the posterior toward the standard-normal prior (so z=0 works at test).
            kld = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).sum(-1).mean()  # scalar
            loss = l1 + self.cfg.kl_weight * kld                        # scalar
            out["kld_loss"] = kld.item()
        out["loss"] = loss.item()
        return loss, out

    # --- inference (rollout in MuJoCo) ----------------------------------------
    def reset(self):
        """Call at the start of each episode rollout."""
        if self.cfg.temporal_ensemble_coeff is not None:
            self._ensembler = TemporalEnsembler(self.cfg.temporal_ensemble_coeff, self.cfg.chunk_size)

    @torch.no_grad()
    def select_action(self, obs: dict) -> Tensor:
        """Pick ONE action for the current MuJoCo step (called every frame).

        Input `obs` (single environment, B=1):
            observation.state         (1, 8)            current joint+gripper pos.
            observation.images.<cam>  (1, 3, 480, 640)  RGB in [0,1], per camera.
        Returns:
            (1, A=8) absolute control target in REAL control units (un-normalized):
            [7 arm joint targets, gripper ctrl 0-255]. No latent here (z=0).
        Strategy: re-query the full k-step chunk every frame, then temporally
        ensemble overlapping predictions so only one smoothed action is emitted.
        """
        self.eval()
        obs = self.normalize_inputs(obs)             # standardize state + images
        actions_hat, _, _ = self.model(obs)          # (1,k,A) normalized; z=0 in eval
        actions = self.unnormalize_action(actions_hat)  # (1,k,A) back to control units
        if self._ensembler is None:
            self.reset()
        return self._ensembler.update(actions)       # (1,A)  one ensembled action
