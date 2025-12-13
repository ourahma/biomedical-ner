import torch
import torch.nn as nn


class CharacterBiLSTM(nn.Module):
    """
    Bi-LSTM au niveau caractère pour capturer les dépendances globales
    """
    def __init__(self, char_vocab_size, embedding_dim=100, hidden_dim=50):
        super(CharacterBiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(char_vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, 
            num_layers=1, bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.5)
        self.output_dim = hidden_dim * 2  # Bidirectionnel
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, word_len)
        batch_size, seq_len, word_len = x.shape
        
        # Reshape pour traiter tous les mots
        x = x.view(batch_size * seq_len, word_len)
        
        # Embedding des caractères
        x = self.embedding(x)  # (batch*seq, word_len, emb_dim)
        
        # Bi-LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Prendre les derniers états forward et backward
        forward_last = hidden[-2, :, :]  # Avant-dernier pour forward
        backward_last = hidden[-1, :, :]  # Dernier pour backward
        
        # Concaténer
        x = torch.cat([forward_last, backward_last], dim=1)
        x = self.dropout(x)
        
        # Reshape pour récupérer la dimension séquence
        x = x.view(batch_size, seq_len, -1)
        
        return x
    
    
