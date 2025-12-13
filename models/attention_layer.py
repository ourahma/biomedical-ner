import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    """
    Mécanisme d'attention entre Bi-LSTM et CRF
    Utilise la distance de Manhattan comme décrit dans l'article
    """
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
    def forward(self, h):
        # h shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _ = h.shape
        
        # Calcul de la distance de Manhattan entre chaque paire de tokens
        # Version vectorisée pour plus d'efficacité
        h_expanded1 = h.unsqueeze(2)  # (batch, seq, 1, hidden)
        h_expanded2 = h.unsqueeze(1)  # (batch, 1, seq, hidden)
        
        # Distance de Manhattan
        manhattan_dist = torch.sum(torch.abs(h_expanded1 - h_expanded2), dim=3)  # (batch, seq, seq)
        attention_scores = -manhattan_dist
        
        # Softmax sur la dernière dimension
        attention_weights = torch.softmax(attention_scores, dim=2)
        
        # Contexte pondéré
        context = torch.bmm(attention_weights, h)
        
        # Concaténation avec l'original
        output = torch.cat([h, context], dim=2)
        
        return output
    
