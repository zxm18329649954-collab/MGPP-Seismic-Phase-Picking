import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import seisbench.generate as sbg
import pytorch_lightning as pl
from typing import Optional, Tuple, List, Dict

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("⚠️  Warning: mamba-ssm not installed. Please install with:")
    print("   pip install mamba-ssm causal-conv1d")
    print("   Falling back to placeholder implementation.")

    class Mamba(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError(
                "Mamba-SSM is not installed. Please install with:\n"
                "  pip install mamba-ssm causal-conv1d\n"
                "Note: This requires CUDA and may take several minutes to compile."
            )
        def forward(self, x):
            pass

phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}
def vector_cross_entropy(y_pred, y_true, eps=1e-5):
    h = y_true * torch.log(y_pred + eps)
    if y_pred.ndim == 3:
        h = h.mean(-1).sum(-1)
    else:
        h = h.sum(-1)
    h = h.mean()
    return -h

def focal_cross_entropy(y_pred, y_true, gamma=2.0, alpha=0.25, eps=1e-5):
    ce = -y_true * torch.log(y_pred + eps)

    p_t = (y_pred * y_true).sum(dim=1, keepdim=True)  # (B, 1, L)
    focal_weight = (1 - p_t) ** gamma

    focal_ce = focal_weight * ce

    class_weights = torch.tensor([alpha, alpha, 1-alpha], device=y_pred.device).view(1, 3, 1)
    weighted_focal_ce = class_weights * focal_ce

    if y_pred.ndim == 3:
        loss = weighted_focal_ce.mean(-1).sum(-1)
    else:
        loss = weighted_focal_ce.sum(-1)

    return loss.mean()


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) * dilation // 2

        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, dilation=dilation, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x
class LocalWindowAttention(nn.Module):

    def __init__(self, dim, num_heads=4, window_size=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape

        pad_len = (self.window_size - L % self.window_size) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        _, L_padded, _ = x.shape
        num_windows = L_padded // self.window_size

        x = x.view(B, num_windows, self.window_size, D)

        qkv = self.qkv(x).reshape(B, num_windows, self.window_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(2, 3).reshape(B, num_windows, self.window_size, D)
        x = x.view(B, L_padded, D)
        x = self.proj(x)

        if pad_len > 0:
            x = x[:, :L, :]

        return x

class CAPE(nn.Module):

    def __init__(self, patch_size, in_channels, embed_dim, threshold=0.5):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.threshold = threshold

        self.mixed_proj = nn.Linear(patch_size * in_channels, embed_dim)
        self.independent_proj = nn.Linear(patch_size, embed_dim)
        self.channel_fusion = nn.Linear(embed_dim * in_channels, embed_dim)

    def compute_channel_correlation(self, x):
        B, C, L = x.shape
        x_centered = x - x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-8
        x_normalized = x_centered / std
        corr = torch.bmm(x_normalized, x_normalized.transpose(1, 2)) / L
        return corr
    def forward(self, x, return_strategy=False):
        B, C, L = x.shape

        corr = self.compute_channel_correlation(x)
        mask = torch.triu(torch.ones(C, C, device=x.device), diagonal=1).bool()
        avg_corr = corr[:, mask].abs().mean(dim=-1)
        use_mixed = avg_corr > self.threshold

        N = L // self.patch_size
        x_patches = x[:, :, :N * self.patch_size].view(B, C, N, self.patch_size)

        x_mixed = x_patches.permute(0, 2, 1, 3).reshape(B, N, C * self.patch_size)
        embed_mixed = self.mixed_proj(x_mixed)

        x_indep = x_patches.permute(0, 1, 2, 3).reshape(B * C, N, self.patch_size)
        embed_indep = self.independent_proj(x_indep)
        embed_indep = embed_indep.view(B, C, N, self.embed_dim)
        embed_indep = embed_indep.permute(0, 2, 1, 3).reshape(B, N, C * self.embed_dim)
        embed_indep = self.channel_fusion(embed_indep)

        use_mixed = use_mixed.view(B, 1, 1).expand(-1, N, self.embed_dim)
        embeddings = torch.where(use_mixed, embed_mixed, embed_indep)

        if return_strategy:
            return embeddings, avg_corr > self.threshold
        return embeddings
class MultiGranularityPartitioning(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        patch_sizes: Tuple[int, ...] = (32, 64, 128),
        cape_threshold: float = 0.5,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_sizes = patch_sizes
        self.num_granularities = len(patch_sizes)

        self.cape_modules = nn.ModuleList([
            CAPE(p, in_channels, embed_dim, cape_threshold)
            for p in patch_sizes
        ])

        self.pos_encodings = nn.ParameterList([
            nn.Parameter(torch.randn(1, max_seq_len // p, embed_dim) * 0.02)
            for p in patch_sizes
        ])

    def forward(self, x):
        embeddings = []
        for i, (cape, pos_enc) in enumerate(zip(self.cape_modules, self.pos_encodings)):
            embed = cape(x)
            N = embed.shape[1]
            embed = embed + pos_enc[:, :N, :]
            embeddings.append(embed)
        return embeddings

class GranularityBranch(nn.Module):

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 2, 4),
        window_size: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilations[i] if i < len(dilations) else dilations[-1]
            self.conv_layers.append(
                DepthwiseSeparableConv(embed_dim, embed_dim, kernel_size, dilation)
            )

        self.residual_proj = nn.Identity()

        self.local_attn = LocalWindowAttention(
            embed_dim, num_heads, window_size, dropout
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        for conv in self.conv_layers:
            x = conv(x) + x
        attn_out = self.local_attn(x)
        x = self.attn_norm(x + self.dropout(attn_out))
        return x
class EnhancedMultiGranularityFeatureExtraction(nn.Module):

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.fine_branch = GranularityBranch(
            embed_dim, num_layers, kernel_size=3,
            dilations=(1, 2, 4), window_size=8,
            num_heads=num_heads, dropout=dropout
        )

        self.medium_branch = GranularityBranch(
            embed_dim, num_layers, kernel_size=5,
            dilations=(1, 2, 4), window_size=16,
            num_heads=num_heads, dropout=dropout
        )

        self.coarse_branch = GranularityBranch(
            embed_dim, num_layers, kernel_size=7,
            dilations=(1, 2, 4), window_size=32,
            num_heads=num_heads, dropout=dropout
        )

        self.branches = [self.fine_branch, self.medium_branch, self.coarse_branch]

    def forward(self, embeddings):
        features = []
        for embed, branch in zip(embeddings, self.branches):
            feat = branch(embed)
            features.append(feat)
        return features

class MambaWrapper(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise RuntimeError(
                "Mamba-SSM is not installed. Please install with:\n"
                "  pip install mamba-ssm causal-conv1d\n"
                "Note: This requires CUDA and may take several minutes to compile."
            )
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        y = self.mamba(x)
        return y, None

class CrossGranularitySSM(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        num_rounds: int = 2,
        dropout: float = 0.1,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_rounds = num_rounds
        self.ssm_fine = MambaWrapper(d_model, d_state, expand=expand)
        self.ssm_medium = MambaWrapper(d_model, d_state, expand=expand)
        self.ssm_coarse = MambaWrapper(d_model, d_state, expand=expand)
        self.cross_12 = nn.Linear(d_model, d_model)
        self.cross_21 = nn.Linear(d_model, d_model)
        self.cross_23 = nn.Linear(d_model, d_model)
        self.cross_32 = nn.Linear(d_model, d_model)
        self.gate_12 = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.gate_21 = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.gate_23 = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.gate_32 = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.lambda_12 = nn.Parameter(torch.tensor(0.1))
        self.lambda_21 = nn.Parameter(torch.tensor(0.1))
        self.lambda_23 = nn.Parameter(torch.tensor(0.1))
        self.lambda_32 = nn.Parameter(torch.tensor(0.1))

        self.granularity_bias = nn.Parameter(torch.tensor([0.2, 0.0, -0.2]))
        self.fusion_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )
        self.norm_fine = nn.LayerNorm(d_model)
        self.norm_medium = nn.LayerNorm(d_model)
        self.norm_coarse = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
    def resample(self, x, target_len):
        if x.shape[1] == target_len:
            return x
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        x = x.transpose(1, 2)
        return x
    def forward(self, features):
        F_fine, F_medium, F_coarse = features
        N_fine, N_medium, N_coarse = F_fine.shape[1], F_medium.shape[1], F_coarse.shape[1]

        for r in range(self.num_rounds):
            Y_fine, _ = self.ssm_fine(self.norm_fine(F_fine))
            Y_medium, _ = self.ssm_medium(self.norm_medium(F_medium))
            Y_coarse, _ = self.ssm_coarse(self.norm_coarse(F_coarse))
            granularity
            bias
            if r < self.num_rounds - 1:
                F_fine_global = F_fine.mean(dim=1)
                F_medium_global = F_medium.mean(dim=1)
                F_coarse_global = F_coarse.mean(dim=1)

                F_medium_to_fine = self.resample(Y_medium, N_fine)
                cross_12 = self.cross_12(F_medium_to_fine)
                gate_12 = self.gate_12(torch.cat([
                    F_fine_global.unsqueeze(1).expand(-1, N_fine, -1),
                    cross_12
                ], dim=-1))
                F_fine = Y_fine + F_fine + self.lambda_12 * gate_12 * cross_12

                F_fine_to_medium = self.resample(Y_fine, N_medium)
                F_coarse_to_medium = self.resample(Y_coarse, N_medium)
                cross_21 = self.cross_21(F_fine_to_medium)
                cross_23 = self.cross_23(F_coarse_to_medium)
                gate_21 = self.gate_21(torch.cat([
                    F_medium_global.unsqueeze(1).expand(-1, N_medium, -1),
                    cross_21
                ], dim=-1))
                gate_23 = self.gate_23(torch.cat([
                    F_medium_global.unsqueeze(1).expand(-1, N_medium, -1),
                    cross_23
                ], dim=-1))
                F_medium = Y_medium + F_medium + self.lambda_21 * gate_21 * cross_21 + self.lambda_23 * gate_23 * cross_23

                F_medium_to_coarse = self.resample(Y_medium, N_coarse)
                cross_32 = self.cross_32(F_medium_to_coarse)
                gate_32 = self.gate_32(torch.cat([
                    F_coarse_global.unsqueeze(1).expand(-1, N_coarse, -1),
                    cross_32
                ], dim=-1))
                F_coarse = Y_coarse + F_coarse + self.lambda_32 * gate_32 * cross_32
            else:
                F_fine = Y_fine + F_fine
                F_medium = Y_medium + F_medium
                F_coarse = Y_coarse + F_coarse

        F_medium_up = self.resample(F_medium, N_fine)
        F_coarse_up = self.resample(F_coarse, N_fine)

        scores = torch.stack([
            self.fusion_attention(F_fine.mean(dim=1)),
            self.fusion_attention(F_medium_up.mean(dim=1)),
            self.fusion_attention(F_coarse_up.mean(dim=1))
        ], dim=1)
        scores = scores + self.granularity_bias.view(1, 3, 1)

        alphas = F.softmax(scores, dim=1)

        output = (alphas[:, 0:1, :] * F_fine +
                  alphas[:, 1:2, :] * F_medium_up +
                  alphas[:, 2:3, :] * F_coarse_up)

        return output
class ResidualConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = F.gelu(out)
        return out
class DeepPhaseHead(nn.Module):

    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels // 2

        self.layer1 = ResidualConvBlock(in_channels, hidden_channels, kernel_size=3)
        self.layer2 = ResidualConvBlock(hidden_channels, hidden_channels, kernel_size=3)
        self.layer3 = ResidualConvBlock(hidden_channels, hidden_channels // 2, kernel_size=3)
        self.layer4 = ResidualConvBlock(hidden_channels // 2, hidden_channels // 2, kernel_size=3)

        self.output = nn.Conv1d(hidden_channels // 2, 1, kernel_size=1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output(x)
        return x
class DeepSeparatedDecoder(nn.Module):

    def __init__(
        self,
        embed_dim: int = 128,
        target_length: int = 3001,
    ):
        super().__init__()
        self.target_length = target_length

        self.up1 = nn.ConvTranspose1d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv1d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(embed_dim // 2)

        self.up2 = nn.ConvTranspose1d(embed_dim // 2, embed_dim // 4, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv1d(embed_dim // 4, embed_dim // 4, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(embed_dim // 4)

        self.up3 = nn.ConvTranspose1d(embed_dim // 4, embed_dim // 8, kernel_size=8, stride=4, padding=2)
        self.conv3 = nn.Conv1d(embed_dim // 8, embed_dim // 8, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm1d(embed_dim // 8)

        head_in_channels = embed_dim // 8
        self.p_head = DeepPhaseHead(head_in_channels, hidden_channels=head_in_channels)
        self.s_head = DeepPhaseHead(head_in_channels, hidden_channels=head_in_channels)
        self.noise_head = nn.Conv1d(head_in_channels, 1, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.up1(x)
        x = self.conv1(F.gelu(x))
        x = self.norm1(x)
        x = self.up2(x)
        x = self.conv2(F.gelu(x))
        x = self.norm2(x)
        x = self.up3(x)
        x = self.conv3(F.gelu(x))
        x = self.norm3(x)
        if x.shape[2] != self.target_length:
            x = F.interpolate(x, size=self.target_length, mode='linear', align_corners=False)
        p_out = self.p_head(x)
        s_out = self.s_head(x)
        noise_out = self.noise_head(x)

        out = torch.cat([p_out, s_out, noise_out], dim=1)
        out = F.softmax(out, dim=1)
        return out
class MGPP(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        in_samples: int = 3001,
        embed_dim: int = 48,
        patch_sizes: Tuple[int, ...] = (32, 64, 128),
        num_feature_layers: int = 2,
        d_state: int = 16,
        num_ssm_rounds: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        cape_threshold: float = 0.5,
        ssm_expand: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_samples = in_samples

        self.partitioning = MultiGranularityPartitioning(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_sizes=patch_sizes,
            cape_threshold=cape_threshold,
            max_seq_len=in_samples,
        )
        self.feature_extraction = EnhancedMultiGranularityFeatureExtraction(
            embed_dim=embed_dim,
            num_layers=num_feature_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cg_ssm = CrossGranularitySSM(
            d_model=embed_dim,
            d_state=d_state,
            num_rounds=num_ssm_rounds,
            dropout=dropout,
            expand=ssm_expand,
        )
        self.decoder = DeepSeparatedDecoder(
            embed_dim=embed_dim,
            target_length=in_samples,
        )
    def forward(self, x):
        embeddings = self.partitioning(x)
        features = self.feature_extraction(embeddings)
        fused = self.cg_ssm(features)
        out = self.decoder(fused)
        return out
class MGPP_PISDL(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        sigma: int = 20,
        in_channels: int = 3,
        in_samples: int = 3001,
        embed_dim: int = 48,
        patch_sizes: Tuple[int, ...] = (32, 64, 128),
        num_feature_layers: int = 2,
        d_state: int = 16,
        num_ssm_rounds: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        cape_threshold: float = 0.5,
        sample_boundaries: Tuple[Optional[int], Optional[int]] = (None, None),
        ssm_expand: int = 2,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.6,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.sigma = sigma
        self.in_samples = in_samples
        self.sample_boundaries = sample_boundaries
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.model = MGPP(
            in_channels=in_channels,
            in_samples=in_samples,
            embed_dim=embed_dim,
            patch_sizes=patch_sizes,
            num_feature_layers=num_feature_layers,
            d_state=d_state,
            num_ssm_rounds=num_ssm_rounds,
            num_heads=num_heads,
            dropout=dropout,
            cape_threshold=cape_threshold,
            ssm_expand=ssm_expand,
        )
    def forward(self, x):
        return self.model(x)
    def compute_loss(self, y_pred, y_true):
        if self.use_focal_loss:
            return focal_cross_entropy(y_pred, y_true,
                                       gamma=self.focal_gamma,
                                       alpha=self.focal_alpha)
        else:
            return vector_cross_entropy(y_pred, y_true)

    def shared_step(self, batch):
        x = batch["X"]
        y_true = batch["y"]
        y_pred = self.model(x)
        loss = self.compute_loss(y_pred, y_true)
        return loss
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }
    def get_augmentations(self):
        return [
            sbg.OneOf(
                [
                    sbg.WindowAroundSample(
                        list(phase_dict.keys()),
                        samples_before=3000,
                        windowlen=6000,
                        selection="random",
                        strategy="variable",
                    ),
                    sbg.NullAugmentation(),
                ],
                probabilities=[2, 1],
            ),
            sbg.RandomWindow(
                low=self.sample_boundaries[0],
                high=self.sample_boundaries[1],
                windowlen=self.in_samples,
                strategy="pad",
            ),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
            sbg.ProbabilisticLabeller(
                label_columns=phase_dict, sigma=self.sigma, dim=0
            ),
        ]
    def get_train_augmentations(self):


    def get_val_augmentations(self):


    def get_eval_augmentations(self):


    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        x = batch["X"]
        pred = self.model(x)
        return pred


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MGPP(in_channels=3, in_samples=3001)
    x = torch.randn(2, 3, 3001)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model parameters: {count_parameters(model):,}")
