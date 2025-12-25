import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np
from torchcrf import CRF
import torch



class CharCNN(nn.Module):
    def __init__(
        self,
        char_vocab_size: int,
        char_embed_dim: int = 30,
        num_filters: int = 32,
        kernel_sizes: List[int] = [3, 5, 7],
    ):
        super().__init__()

        self.char_embedding = nn.Embedding(
            char_vocab_size, char_embed_dim, padding_idx=0
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=char_embed_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2
            )
            for k in kernel_sizes
        ])

    def forward(self, char_ids):
        # char_ids: [B, T, W]
        B, T, W = char_ids.size()

        char_ids = char_ids.view(B * T, W)
        emb = self.char_embedding(char_ids)              # [B*T, W, C]
        emb = emb.transpose(1, 2)                         # [B*T, C, W]

        conv_feats = []
        for conv in self.convs:
            x = torch.relu(conv(emb))                     # [B*T, F, W]
            x, _ = torch.max(x, dim=2)                    # [B*T, F]
            conv_feats.append(x)

        out = torch.cat(conv_feats, dim=1)                # [B*T, F*len(K)]
        return out.view(B, T, -1)

class CharBiLSTM(nn.Module):
    def __init__(
        self,
        char_vocab_size: int,
        char_embed_dim: int = 30,
        hidden_dim: int = 50,
    ):
        super().__init__()

        self.char_embedding = nn.Embedding(
            char_vocab_size, char_embed_dim, padding_idx=0
        )

        self.lstm = nn.LSTM(
            char_embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, char_ids):
        # char_ids: [B, T, W]
        B, T, W = char_ids.size()

        char_ids = char_ids.view(B * T, W)
        emb = self.char_embedding(char_ids)               # [B*T, W, C]

        _, (h, _) = self.lstm(emb)

        # concat last forward + backward
        out = torch.cat([h[0], h[1]], dim=1)               # [B*T, 2H]
        return out.view(B, T, -1)
class TokenAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H, mask):
        # H: [B, T, H]
        scores = self.v(torch.tanh(self.W(H))).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~mask, -1e9)

        alpha = torch.softmax(scores, dim=1).unsqueeze(-1) # [B, T, 1]
        return H * alpha

class CombinatorialNER(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        char_vocab_size: int,
        tag_to_idx: Dict[str, int],
        use_char_cnn: bool = True,
        use_char_lstm: bool = True,
        use_attention: bool = True,
        use_fc_fusion: bool = True,
        pretrained_embeddings: Optional[np.ndarray] = None,
        word_embed_dim: int = 200,
        lstm_hidden_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.use_char_cnn = use_char_cnn
        self.use_char_lstm = use_char_lstm
        self.use_attention = use_attention
        self.use_fc_fusion = use_fc_fusion

        # Word embeddings
        if pretrained_embeddings is not None:
            self.word_embedding = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_embeddings, dtype=torch.float),
                padding_idx=0,
                freeze=False
            )
        else:
            self.word_embedding = nn.Embedding(
                vocab_size, word_embed_dim, padding_idx=0
            )

        # Char encoders
        if self.use_char_cnn:
            self.char_cnn = CharCNN(char_vocab_size)
        if self.use_char_lstm:
            self.char_lstm = CharBiLSTM(char_vocab_size)

        # Calculer dimension combinée dynamique
        char_dim = 0
        if self.use_char_cnn:
            char_dim += 32 * 3  # CNN (3 kernels)
        if self.use_char_lstm:
            char_dim += 100      # BiLSTM hidden_dim*2

        combined_dim = word_embed_dim + char_dim

        # Fusion fully connected
        if self.use_fc_fusion:
            self.fusion = nn.Sequential(
                nn.Linear(combined_dim, 200),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            lstm_input_dim = 200
        else:
            lstm_input_dim = combined_dim

        # Context BiLSTM
        self.context_lstm = nn.LSTM(
            lstm_input_dim,
            lstm_hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )

        # Attention
        if self.use_attention:
            self.attention_layer = TokenAttention(lstm_hidden_dim)

        # Emissions
        self.emission = nn.Linear(lstm_hidden_dim, len(tag_to_idx))

        # CRF
        self.crf = CRF(len(tag_to_idx))

    def forward(self, word_ids, char_ids, mask, tags=None):
        word_emb = self.word_embedding(word_ids)

        char_embs = []
        if self.use_char_cnn:
            char_embs.append(self.char_cnn(char_ids))
        if self.use_char_lstm:
            char_embs.append(self.char_lstm(char_ids))

        combined = torch.cat([word_emb] + char_embs, dim=-1)

        if self.use_fc_fusion:
            combined = self.fusion(combined)

        lstm_out, _ = self.context_lstm(combined)

        if self.use_attention:
            lstm_out = self.attention_layer(lstm_out, mask)

        emissions = self.emission(lstm_out)

        # CRF ATTEND (T, B, C)
        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1)

        if tags is not None:
            tags = tags.transpose(0, 1)
            return -self.crf(emissions, tags, mask=mask).mean()
        else:
            return self.crf._viterbi_decode(emissions, mask=mask)

