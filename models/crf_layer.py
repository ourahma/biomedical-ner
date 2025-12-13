import torch
import torch.nn as nn

class CRFLayer(nn.Module):
    """
    Couche CRF complète avec matrice de transitions et algorithme de Viterbi
    """
    def __init__(self, num_tags, padding_idx=0):
        super(CRFLayer, self).__init__()
        self.num_tags = num_tags
        self.padding_idx = padding_idx
        
        # Matrice de transition: transition[i, j] = score de transition de tag i à tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Contraintes: transitions depuis et vers le padding sont très improbables
        self.transitions.data[padding_idx, :] = -10000
        self.transitions.data[:, padding_idx] = -10000
        
        # Émission sera gérée par une couche linéaire séparée
    
    def forward(self, features, labels=None, mask=None):
        """
        features: (batch_size, seq_len, feature_dim) - sorties du Bi-LSTM
        labels: (batch_size, seq_len) - labels ground truth (optionnel)
        mask: (batch_size, seq_len) - masque pour ignorer le padding
        """
        batch_size, seq_len, feature_dim = features.shape
        
        # Si pas de masque fourni, créer un masque basé sur les labels non-padding
        if mask is None and labels is not None:
            mask = (labels != self.padding_idx).float()
        elif mask is None:
            mask = torch.ones(batch_size, seq_len, device=features.device)
        
        # Calcul des scores d'émission (sera fait dans le modèle principal)
        # Ici on suppose que features sont déjà les scores d'émission
        emissions = features
        
        if labels is not None:
            # Mode entraînement: calculer la log-likelihood négative
            return self._compute_negative_log_likelihood(emissions, labels, mask)
        else:
            # Mode inférence: décodage Viterbi
            return self._viterbi_decode(emissions, mask)
    
    def _compute_negative_log_likelihood(self, emissions, labels, mask):
        """
        Calcule la log-likelihood négative pour l'entraînement
        """
        batch_size, seq_len, num_tags = emissions.shape
        
        # Score des séquences d'or
        gold_score = self._score_sequence(emissions, labels, mask)
        
        # Score de toutes les séquences possibles (log-sum-exp)
        total_score = self._compute_log_partition_function(emissions, mask)
        
        # Log-likelihood négative
        nll = total_score - gold_score
        
        # Moyenne sur le batch
        return nll.mean()
    
    def _score_sequence(self, emissions, labels, mask):
        """
        Calcule le score d'une séquence de labels
        """
        batch_size, seq_len, num_tags = emissions.shape
        
        # Score des émissions
        emissions_score = emissions.gather(2, labels.unsqueeze(2)).squeeze(2)  # (batch, seq)
        emissions_score = emissions_score * mask
        seq_emission_score = emissions_score.sum(dim=1)  # (batch,)
        
        # Score des transitions
        # Pour chaque transition entre labels[i] et labels[i+1]
        labels_expanded = labels.unsqueeze(2)  # (batch, seq, 1)
        
        # Récupérer les scores de transition
        # transition_score[i] = transition[labels[i], labels[i+1]]
        transition_scores = self.transitions[labels[:, :-1], labels[:, 1:]]  # (batch, seq-1)
        
        # Appliquer le masque (on ignore les transitions impliquant du padding)
        transition_mask = mask[:, :-1] * mask[:, 1:]  # (batch, seq-1)
        transition_scores = transition_scores * transition_mask
        seq_transition_score = transition_scores.sum(dim=1)  # (batch,)
        
        # Score total
        total_score = seq_emission_score + seq_transition_score
        
        return total_score
    
    def _compute_log_partition_function(self, emissions, mask):
        """
        Calcule la fonction de partition (log-sum-exp sur toutes les séquences)
        avec l'algorithme forward
        """
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialisation de l'algorithme forward
        # alpha[i] = log-sum-exp des scores de toutes les séquences terminant au tag i à la position t
        alpha = emissions[:, 0, :]  # (batch, num_tags)
        
        # Itération forward
        for t in range(1, seq_len):
            # alpha_prev expand: (batch, num_tags, 1)
            alpha_prev = alpha.unsqueeze(2)
            
            # transitions expand: (1, num_tags, num_tags)
            transitions = self.transitions.unsqueeze(0)
            
            # emissions expand: (batch, 1, num_tags)
            emission_t = emissions[:, t, :].unsqueeze(1)
            
            # score = alpha_prev + transitions + emission_t
            # (batch, num_tags, num_tags)
            scores = alpha_prev + transitions + emission_t
            
            # Nouvel alpha: log-sum-exp sur les tags précédents
            alpha = torch.logsumexp(scores, dim=1)  # (batch, num_tags)
            
            # Appliquer le masque
            mask_t = mask[:, t].unsqueeze(1)  # (batch, 1)
            alpha = alpha * mask_t + alpha_prev.squeeze(2) * (1 - mask_t)
        
        # Score total: log-sum-exp sur les tags finaux
        total_score = torch.logsumexp(alpha, dim=1)  # (batch,)
        
        return total_score
    
    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape

        viterbi_scores = emissions.new_full((batch_size, num_tags), -1e4)
        viterbi_scores[:, :] = emissions[:, 0, :]
        backpointers = emissions.new_zeros((batch_size, seq_len, num_tags), dtype=torch.long)

        for t in range(1, seq_len):
            broadcast_score = viterbi_scores.unsqueeze(2)          # (batch, num_tags, 1)
            broadcast_trans = self.transitions.unsqueeze(0)        # (1, num_tags, num_tags)
            broadcast_emission = emissions[:, t].unsqueeze(1)      # (batch, 1, num_tags)

            scores = broadcast_score + broadcast_trans + broadcast_emission
            max_scores, best_tags = scores.max(dim=1)

            mask_t = mask[:, t].unsqueeze(1)
            viterbi_scores = mask_t * max_scores + (1 - mask_t) * viterbi_scores
            backpointers[:, t] = best_tags

        best_tags = viterbi_scores.argmax(dim=1)

        best_paths = emissions.new_zeros((batch_size, seq_len), dtype=torch.long)
        best_paths[:, -1] = best_tags

        for t in range(seq_len - 2, -1, -1):
            best_paths[:, t] = backpointers[
                torch.arange(batch_size), t + 1, best_paths[:, t + 1]
            ]

        return best_paths

