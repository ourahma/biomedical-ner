from torch.utils.data import Dataset, DataLoader
import re
import torch
from utils.creation_vocabulaire import create_vocab,create_char_vocab,preprocess_tokens,create_char_sequences,create_tag_mapping

class NERDataset(Dataset):
    def __init__(self, sentences, vocab, char_vocab, tag_to_idx, max_seq_len=100):
        """
        Dataset pour NER - Supporte deux formats:
        1. Format JNLPBA: Liste de tuples (tokens, labels) où tokens et labels sont des listes
        2. Format NCBI: Liste de listes de tuples (token, label)
        """
        self.data = []
        self.max_seq_len = max_seq_len
        
        # Détecter le format
        if not sentences:
            return
            
        first_item = sentences[0]
        
        for sentence in sentences:
            if isinstance(sentence, tuple) and len(sentence) == 2:
                # Format JNLPBA: (tokens, labels)
                tokens, labels = sentence
                
            elif isinstance(sentence, list):
                # Format NCBI original: liste de (token, label)
                tokens = []
                labels = []
                for item in sentence:
                    if isinstance(item, tuple) and len(item) == 2:
                        token, label = item
                        tokens.append(token)
                        labels.append(label)
                    else:
                        print(f"Élément non tuple dans liste: {type(item)}")
                        continue
            else:
                print(f"Format inconnu: {type(sentence)}")
                continue
            
            # Vérifier que nous avons des données
            if not tokens or not labels or len(tokens) != len(labels):
                print(f"Phrase invalide: tokens={len(tokens)}, labels={len(labels)}")
                continue
            
            # Limiter la longueur de la séquence
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
                labels = labels[:max_seq_len]
            
            # Convertir tokens en IDs
            word_ids = preprocess_tokens(tokens, vocab)
            
            # Créer séquences de caractères
            char_seqs = create_char_sequences(tokens, char_vocab)
            
            # Convertir labels en IDs
            label_ids = [tag_to_idx.get(tag, 0) for tag in labels]  # 0 = <PAD>
            
            # Padding
            padding_len = max_seq_len - len(word_ids)
            if padding_len > 0:
                word_ids = word_ids + [vocab['<PAD>']] * padding_len
                char_seqs = char_seqs + [[char_vocab['<PAD>']] * 20] * padding_len
                label_ids = label_ids + [0] * padding_len  # 0 = <PAD> pour les labels
            
            self.data.append((word_ids, char_seqs, label_ids, len(tokens)))
        
        print(f"Dataset créé: {len(self.data)} phrases valides")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        word_ids, char_seqs, label_ids, actual_len = self.data[idx]
        return (
            torch.LongTensor(word_ids),
            torch.LongTensor(char_seqs),
            torch.LongTensor(label_ids),
            actual_len
        )