import torch
import torch.nn as nn
import models

from models.character_cnn import CharacterCNN
from models.character_bilstm import CharacterBiLSTM
from models.attention_layer import AttentionLayer
from models.crf_layer import CRFLayer



class BiomedicalNERModel(nn.Module):
    """
    Modèle complet de NER biomédical avec combinatorial feature embedding
    Architecture: word_emb + char_cnn + char_bilstm → FC → Bi-LSTM → Attention → Emission → CRF
    """
    def __init__(self, word_vocab_size, char_vocab_size, label_vocab_size,
                 word_emb_dim=200, char_emb_dim=100, 
                 char_cnn_filters=[3, 5, 7], char_cnn_filters_num=32,
                 char_lstm_hidden=50, lstm_hidden=200,
                 dropout=0.5, use_attention=True):
        
        super(BiomedicalNERModel, self).__init__()
        
        # 1. Embeddings de mots
        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=0)
        
        # 2. Embeddings caractère: CNN
        self.char_cnn = CharacterCNN(
            char_vocab_size, char_emb_dim, 
            char_cnn_filters, char_cnn_filters_num
        )
        
        # 3. Embeddings caractère: Bi-LSTM
        self.char_bilstm = CharacterBiLSTM(
            char_vocab_size, char_emb_dim, 
            char_lstm_hidden
        )
        
        # 4. Couche fully-connected pour combiner les features
        combined_dim = word_emb_dim + self.char_cnn.output_dim + self.char_bilstm.output_dim
        self.fc_combine = nn.Sequential(
            nn.Linear(combined_dim, 200),
            nn.ReLU() if dropout > 0 else nn.Identity(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        
        # 5. Bi-LSTM principal
        self.bilstm = nn.LSTM(
            200, lstm_hidden // 2,
            num_layers=1, bidirectional=True,
            batch_first=True, dropout=dropout if dropout > 0 else 0
        )
        
        # 6. Mécanisme d'attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionLayer(lstm_hidden)
            crf_input_dim = lstm_hidden * 2
        else:
            crf_input_dim = lstm_hidden
        
        # 7. Couche d'émission (pour convertir les features en scores)
        self.emission = nn.Linear(crf_input_dim, label_vocab_size)
        
        # 8. Couche CRF (la vraie!)
        self.crf = CRFLayer(label_vocab_size, padding_idx=0)
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation des poids du modèle"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                if 'lstm' in name:
                    # Initialisation spéciale pour LSTM
                    for i in range(4):
                        nn.init.orthogonal_(param.data[i*param.shape[1]//4:(i+1)*param.shape[1]//4, :])
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, words, chars, labels=None, mask=None):
        batch_size, seq_len = words.shape
        
        # Si pas de masque fourni, créer un masque basé sur les mots non-padding
        if mask is None:
            mask = (words != 0).float()  # 0 = <PAD>
        
        # 1. Embeddings de mots
        word_emb = self.word_embedding(words)  # (batch, seq, word_emb_dim)
        
        # 2. Embeddings caractère CNN
        char_cnn_emb = self.char_cnn(chars)  # (batch, seq, cnn_output_dim)
        
        # 3. Embeddings caractère Bi-LSTM
        char_bilstm_emb = self.char_bilstm(chars)  # (batch, seq, bilstm_output_dim)
        
        # 4. Combinaison: Concaténation + FC
        combined = torch.cat([word_emb, char_cnn_emb, char_bilstm_emb], dim=2)
        combined = self.fc_combine(combined)  # (batch, seq, 200)
        
        # 5. Bi-LSTM principal
        lstm_out, _ = self.bilstm(combined)  # (batch, seq, lstm_hidden)
        
        # 6. Attention (si activée)
        if self.use_attention:
            attention_out = self.attention(lstm_out)  # (batch, seq, lstm_hidden*2)
            features = attention_out
        else:
            features = lstm_out
        
        # 7. Scores d'émission
        emissions = self.emission(features)  # (batch, seq, num_tags)
        
        # 8. CRF
        if labels is not None:
            # Mode entraînement: calculer la loss CRF
            crf_loss = self.crf(emissions, labels, mask)
            
            # Pour le débogage, aussi calculer la loss cross-entropy
            ce_loss = self._compute_ce_loss(emissions, labels, mask)
            
            # Retourner les deux pour monitoring
            return crf_loss, ce_loss
        else:
            # Mode inférence: décodage Viterbi
            predictions = self.crf(emissions, mask=mask)
            return predictions
    
    def _compute_ce_loss(self, emissions, labels, mask):
        """Calcule aussi la loss cross-entropy pour monitoring"""
        # Reshape pour la cross-entropy
        batch_size, seq_len, num_tags = emissions.shape
        emissions_reshaped = emissions.view(-1, num_tags)
        labels_reshaped = labels.view(-1)
        mask_reshaped = mask.view(-1)
        
        # Ignorer les positions de padding
        non_pad_indices = mask_reshaped.nonzero().squeeze()
        
        if len(non_pad_indices) > 0:
            ce_loss = nn.CrossEntropyLoss()(
                emissions_reshaped[non_pad_indices],
                labels_reshaped[non_pad_indices]
            )
            return ce_loss
        else:
            return torch.tensor(0.0, device=emissions.device)
    
    def predict(self, words, chars, mask=None):
        """Méthode pour faire des prédictions"""
        self.eval()
        with torch.no_grad():
            if mask is None:
                mask = (words != 0).float()
            return self(words, chars, mask=mask)