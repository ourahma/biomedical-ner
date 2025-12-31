import torch
import torch.nn as nn
from typing import Dict, Optional, List
from torchcrf import CRF
import numpy as np

# ==================== CharCNN ====================
class CharCNN(nn.Module):
    def __init__(self, char_vocab_size: int, char_embed_dim: int = 50,
                 num_filters: int = 32, kernel_sizes: List[int] = [3,5,7], dropout: float = 0.5):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=char_embed_dim, out_channels=num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids):
        B, T, W = char_ids.size()
        char_ids = char_ids.view(B*T, W)
        emb = self.char_embedding(char_ids).transpose(1,2)
        conv_feats = []
        for conv in self.convs:
            x = torch.relu(conv(emb))
            x, _ = torch.max(x, dim=2)
            conv_feats.append(x)
        out = torch.cat(conv_feats, dim=1)
        return self.dropout(out).view(B, T, -1)

# ==================== CharBiLSTM ====================
class CharBiLSTM(nn.Module):
    def __init__(self, char_vocab_size: int, char_embed_dim: int = 50,
                 hidden_dim: int = 50, dropout: float = 0.5):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(char_embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids):
        B, T, W = char_ids.size()          # B=batch, T=seq_len, W=word_len
        char_ids = char_ids.view(B*T, W)   # aplatissement pour passer dans LSTM
        emb = self.char_embedding(char_ids)  # (B*T, W, char_embed_dim)

        output, _ = self.lstm(emb)         # sortie LSTM complète (B*T, W, hidden*2)
        output = self.dropout(output)

        # max pooling sur les caractères pour obtenir un vecteur par mot
        out, _ = torch.max(output, dim=1)  # (B*T, hidden*2)

        return out.view(B, T, -1)          # reshape pour (B, T, hidden*2)



# ==================== Manhattan Attention ====================
class ManhattanAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, 1, bias=False)

    def forward(self, h, mask):
        B, T, D = h.shape
        hi = h.unsqueeze(2).expand(B, T, T, D)
        hj = h.unsqueeze(1).expand(B, T, T, D)
        dist = torch.abs(hi - hj).sum(-1)                # Manhattan distance
        score = -self.W(hj).squeeze(-1) * dist
        score = score.masked_fill(~mask.unsqueeze(1), -1e9)
        alpha = torch.softmax(score, -1)
        ctx = torch.matmul(alpha, h)
        return torch.cat([h, ctx], -1)                  # concat context

# ==================== CombinatorialNER ====================
class CombinatorialNER(nn.Module):
    def __init__(self, vocab_size: int, char_vocab_size: int, tag_to_idx: Dict[str,int],
                 dataset: str = "JNLPBA",
                 use_char_cnn=True, use_char_lstm=True,
                 use_attention=True, use_fc_fusion=True,
                 pretrained_embeddings: Optional[np.ndarray] = None,
                 word_embed_dim: int = 200, lstm_hidden_dim: int = 256,
                 dropout: float = 0.5, use_lstm=True):
        super().__init__()

        self.dataset = dataset
        self.use_char_cnn = use_char_cnn
        self.use_char_lstm = use_char_lstm
        self.lstm_hidden_dim = lstm_hidden_dim 
        self.use_attention = use_attention
        self.use_fc_fusion = use_fc_fusion
        self.use_lstm = use_lstm

        # Word embeddings
        if pretrained_embeddings is not None:
            self.word_embedding = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_embeddings, dtype=torch.float), padding_idx=0, freeze=False
            )
        else:
            self.word_embedding = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)

        # Char encoders
        if self.use_char_cnn:
            cnn_kernels = [3,5,7] if dataset=="JNLPBA" else [2,3,4]
            self.char_cnn = CharCNN(char_vocab_size, char_embed_dim=50,
                                    num_filters=32, kernel_sizes=cnn_kernels, dropout=dropout)
        if self.use_char_lstm:
            self.char_lstm = CharBiLSTM(char_vocab_size, char_embed_dim=50, hidden_dim=50, dropout=dropout)

        # Combined dimension
        char_dim = 0
        if self.use_char_cnn:
            char_dim += 32 * len(cnn_kernels)
        if self.use_char_lstm:
            char_dim += 100  # 50*2

        combined_dim = word_embed_dim + char_dim

        # FC fusion
        if self.use_fc_fusion:
            fusion_out_dim = 200
            if dataset=="NCBI":
                self.fusion = nn.Sequential(
                    nn.Linear(combined_dim, fusion_out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            else:
                self.fusion = nn.Sequential(
                    nn.Linear(combined_dim, fusion_out_dim),
                    nn.Dropout(dropout)
                )
            lstm_input_dim = fusion_out_dim
        else:
            lstm_input_dim = combined_dim

        # Context BiLSTM optionnel
        if self.use_lstm and (self.use_char_cnn or self.use_char_lstm or self.use_attention or self.use_fc_fusion):
            self.context_lstm = nn.LSTM(
                lstm_input_dim,
                lstm_hidden_dim // 2,
                batch_first=True,
                bidirectional=True
            )

            if self.use_attention:
                self.attention_layer = ManhattanAttention(lstm_hidden_dim)
                # Correctement défini ici après attention_layer
                self.attention_projection = nn.Linear(lstm_hidden_dim*2, lstm_hidden_dim)
                lstm_output_dim = lstm_hidden_dim
            else:
                self.attention_layer = None
                lstm_output_dim = lstm_hidden_dim
        else:
            self.context_lstm = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=lstm_input_dim // 2,
                batch_first=True,
                bidirectional=True
            )
            self.attention_layer = None
            lstm_output_dim = lstm_input_dim

        # Emission & CRF
        self.emission = nn.Linear(lstm_output_dim, len(tag_to_idx))
        self.crf = CRF(len(tag_to_idx))

    def forward(self, word_ids, char_ids=None, mask=None, tags=None):
        word_emb = self.word_embedding(word_ids)

        char_embs = []
        if self.use_char_cnn and char_ids is not None:
            char_embs.append(self.char_cnn(char_ids))
        if self.use_char_lstm and char_ids is not None:
            char_embs.append(self.char_lstm(char_ids))
        combined = torch.cat([word_emb] + char_embs, dim=-1) if char_embs else word_emb

        if self.use_fc_fusion:
            combined = self.fusion(combined)

        if self.context_lstm is not None:
            lstm_out, _ = self.context_lstm(combined)
            if self.attention_layer is not None and mask is not None:
                lstm_out = self.attention_layer(lstm_out, mask)
                lstm_out = self.attention_projection(lstm_out)
        else:
            lstm_out = combined

        emissions = self.emission(lstm_out).transpose(0,1)
        mask_t = mask.transpose(0,1) if mask is not None else None

        if tags is not None:
            tags = tags.transpose(0,1)
            return -self.crf(emissions, tags, mask=mask_t).mean()
        else:
            return self.crf.decode(emissions, mask=mask_t)