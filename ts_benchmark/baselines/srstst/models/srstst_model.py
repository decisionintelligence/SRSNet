import torch
from torch import nn

from ts_benchmark.baselines.srstst.layers.Embed import SRS
from ts_benchmark.baselines.srstst.layers.SelfAttention_Family import FullAttention, AttentionLayer
from ts_benchmark.baselines.srstst.layers.Transformer_EncDec import Encoder, EncoderLayer
from ts_benchmark.baselines.srstst.layers.RevIN import RevIN


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class SRSTSTModel(nn.Module):
    def __init__(self, config):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super(SRSTSTModel, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.stride = config.stride

        # selective representation space
        self.patch_embedding = SRS(
            config.d_model, self.patch_len, self.stride, self.seq_len, config.dropout, config.hidden_size
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )

        # Prediction Head
        self.head_nf = config.d_model * int((config.seq_len - self.patch_len) / self.stride + 2)
        self.head = FlattenHead(
            config.enc_in,
            self.head_nf,
            config.pred_len,
            head_dropout=config.dropout,
        )

        self.revin = RevIN(num_features=config.enc_in, affine=config.affine, subtract_last=config.subtract_last)

    def forward(self, x_enc):
        x_enc = self.revin(x_enc, 'norm')
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = self.revin(dec_out, 'denorm')
        return dec_out
