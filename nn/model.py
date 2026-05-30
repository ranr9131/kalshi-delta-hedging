"""
Small transformer encoder for KXBTC15M outcome prediction.

Input  : (B, T, F) — batched sequences of T minutes × F features
Output : (B,)       — logit for P(YES wins at close)

Total params with default config: ~17k. Intentionally small — 7k training
examples can't support a bigger model without overfitting.
"""

import torch
import torch.nn as nn


class TSWinPredictor(nn.Module):
    def __init__(self, n_features: int = 7, d_model: int = 32,
                 n_heads: int = 4, n_layers: int = 2,
                 dim_feedforward: int = 64, max_len: int = 15,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb    = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True, dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head    = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:    (B, T, F)  float
        mask: (B, T)     bool — True where the timestep is VALID
        """
        B, T, _ = x.shape
        h = self.input_proj(x)
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = h + self.pos_emb(positions)
        # nn.TransformerEncoder's src_key_padding_mask treats True as PAD.
        kp_mask = (~mask) if mask is not None else None
        h = self.encoder(h, src_key_padding_mask=kp_mask)
        # Masked mean pool — ignore padded timesteps.
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            pooled = h.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.head(pooled).squeeze(-1)
