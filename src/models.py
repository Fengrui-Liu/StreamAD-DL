import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerEncoder
from src.dlutils import Qnet, Pnet

# Used to initialize the weights of the model.

from src.dlutils import (
    ConvLSTM,
    PositionalEncoding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Simple Multi-Head Self-Attention Model
class Attention(nn.Module):
    def __init__(self, feats: int, n_window: int):
        super(Attention, self).__init__()
        self.name = "Attention"
        self.n_feats = feats
        self.n_window = n_window
        self.n = self.n_feats * self.n_window
        self.atts = nn.MultiheadAttention(
            embed_dim=self.n_feats, num_heads=1, batch_first=True
        )

    def forward(self, x):
        x, _ = self.atts(x, x, x)

        return x


# DAGMM Model (ICLR 18)
class DAGMM(nn.Module):
    def __init__(self, feats: int, n_window: int):
        super(DAGMM, self).__init__()
        self.name = "DAGMM"
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 16
        self.n_latent = 8
        self.n_window = n_window
        self.n = self.n_feats * self.n_window
        self.n_gmm = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden),
            nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden),
            nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Tanh(),
            nn.Linear(self.n_hidden, self.n),
            nn.Sigmoid(),
        )
        self.estimate = nn.Sequential(
            nn.Linear(self.n_latent + 2, self.n_hidden),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(self.n_hidden, self.n_gmm),
            nn.Softmax(dim=1),
        )

    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x - x_hat).norm(2, dim=1) / x.norm(
            2, dim=1
        )
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity

    def forward(self, x):
        # Encode Decoder
        x_shape = x.shape
        x = x.view(-1, self.n)
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)
        # Compute Reconstructoin
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        # Estimate
        gamma = self.estimate(z)
        return z_c, x_hat.view(x_shape), z, gamma.view(x_shape)


# OmniAnomaly Model (KDD 19) without normalization flow.
class OmniAnomaly(nn.Module):
    def __init__(self, feats: int, n_window: int):
        super(OmniAnomaly, self).__init__()
        self.name = "OmniAnomaly"
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 64
        self.n_window = n_window
        self.n = self.n_feats * self.n_window
        self.n_latent = 16
        self.lstm = nn.GRU(
            input_size=self.n_feats,
            hidden_size=self.n_hidden,
            num_layers=2,
            batch_first=True,
        )
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.PReLU(),
            # nn.Flatten(),
            nn.Linear(self.n_hidden, 2 * self.n_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden),
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats),
            nn.Sigmoid(),
        )

    def forward(self, x):

        hidden = nn.init.xavier_normal_(
            torch.zeros(2, x.shape[0], self.n_hidden, dtype=torch.float64)
        ).to(device)
        out, hidden = self.lstm(x, hidden)
        # Encode
        x = self.encoder(out)
        mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = mu + eps * std
        # Decoder
        x = self.decoder(x)
        return x, mu, logvar


# OmniAnomaly (This model is too slow to train, we do not use it temporarily)
# class OmniAnomaly(nn.Module):
#     def __init__(self, feats: int, n_window: int):
#         super(OmniAnomaly, self).__init__()
#         self.name = "OmniAnomaly"
#         self.n_hidden = 64
#         self.n_feats = feats
#         self.z_dim = 16
#         self.dense_dim = 32
#         self.qnet = Qnet(
#             in_dim=self.n_feats,
#             hidden_dim=self.n_hidden,
#             z_dim=self.z_dim,
#             dense_dim=self.dense_dim,
#         ).to(device)
#         self.pnet = Pnet(
#             z_dim=self.z_dim,
#             hidden_dim=self.n_hidden,
#             dense_dim=self.dense_dim,
#             out_dim=self.n_feats,
#         ).to(device)

#     def forward(self, x):
#         z, mu, logvar = self.qnet(x)
#         out = self.pnet(z)
#         return out, mu, logvar


# USAD Model (KDD 20)
class USAD(nn.Module):
    def __init__(self, feats: int, n_window: int):
        super(USAD, self).__init__()
        self.name = "USAD"
        self.n_feats = feats
        self.n_hidden = 16
        self.forward_n = 1
        self.n_latent = 5
        self.n_window = n_window
        self.n = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_latent),
            nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n),
            nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n),
            nn.Sigmoid(),
        )

    def forward(self, g):
        # Encode
        g_shape = g.shape
        z = self.encoder(g)
        # Decoders (Phase 1)
        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)
        # Encode-Decode (Phase 2)
        ae2ae1 = self.decoder2(self.encoder(ae1))

        # Update epoch_n and forward_n
        if self.training:
            self.forward_n += 1

        return ae1.view(g_shape), ae2.view(g_shape), ae2ae1.view(g_shape)


# Basic Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
    def __init__(self, feats: int, n_window: int):
        super(TranAD, self).__init__()
        self.name = "TranAD"
        self.epoch_n = 1
        self.forward_n = 1
        self.batch = 128
        self.n_feats = feats
        self.n_window = n_window
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        # src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src):

        if self.training:
            self.forward_n += 1

        src_shape = src.shape
        tgt = src.clone().detach().requires_grad_(True)
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1.view(src_shape), x2.view(src_shape)
