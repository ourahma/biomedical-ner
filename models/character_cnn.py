import torch
import torch.nn as nn

class CharacterCNN(nn.Module):
    """
    CNN au niveau caractère pour capturer les motifs locaux
    """
    def __init__(self, char_vocab_size, embedding_dim=100, filters=[3, 5, 7], num_filters=32):
        super(CharacterCNN, self).__init__()
        
        self.embedding = nn.Embedding(char_vocab_size, embedding_dim, padding_idx=0)
        
        # Convolutions 1D avec différentes tailles de fenêtre
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding=k//2)
            for k in filters
        ])
        
        self.dropout = nn.Dropout(0.5)
        self.output_dim = num_filters * len(filters)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, word_len)
        batch_size, seq_len, word_len = x.shape
        
        # Reshape pour traiter tous les mots
        x = x.view(batch_size * seq_len, word_len)
        
        # Embedding des caractères
        x = self.embedding(x)  # (batch*seq, word_len, emb_dim)
        x = x.transpose(1, 2)  # (batch*seq, emb_dim, word_len)
        
        # Appliquer les convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))
            conv_out, _ = torch.max(conv_out, dim=2)  # Max pooling global
            conv_outputs.append(conv_out)
        
        # Concaténer les sorties
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        
        # Reshape pour récupérer la dimension séquence
        x = x.view(batch_size, seq_len, -1)
        
        return x


