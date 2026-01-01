
import torch
import pickle
import os
from pathlib import Path
import re
from typing import List, Dict, Tuple

def load_all_components(model_checkpoint_path: str, vocab_dir: str, word2vec_model_path: str = "./word2Vecembeddings/jnlpba_word2vec"):
    """
    Charge tous les composants nécessaires pour Streamlit
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🔧 Chargement des composants...")
    
    # 1. Charger les vocabulaires sauvegardés
    with open(os.path.join(vocab_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    
    with open(os.path.join(vocab_dir, 'char_vocab.pkl'), 'rb') as f:
        char_vocab = pickle.load(f)
    
    with open(os.path.join(vocab_dir, 'tag_to_idx.pkl'), 'rb') as f:
        tag_to_idx = pickle.load(f)
    
    with open(os.path.join(vocab_dir, 'idx_to_tag.pkl'), 'rb') as f:
        idx_to_tag = pickle.load(f)
    
    print(f"✅ Vocabulaires chargés: {len(vocab)} mots, {len(char_vocab)} caractères")
    
    # 2. Charger le Word2Vec si nécessaire
    pretrained_embeddings = None
    if word2vec_model_path and os.path.exists(word2vec_model_path):
        try:
            from gensim.models import Word2Vec
            print(f"📚 Chargement du modèle Word2Vec: {word2vec_model_path}")
            word2vec_model = Word2Vec.load(word2vec_model_path)
            
            # Créer la matrice d'embeddings
            import numpy as np
            embedding_dim = 200
            pretrained_embeddings = np.zeros((len(vocab), embedding_dim))
            
            words_found = 0
            for word, idx in vocab.items():
                if word == '<PAD>':
                    pretrained_embeddings[idx] = np.zeros(embedding_dim)
                elif word == '<UNK>':
                    pretrained_embeddings[idx] = np.random.normal(scale=0.1, size=(embedding_dim,))
                elif word == '<NUM>':
                    pretrained_embeddings[idx] = np.random.normal(scale=0.05, size=(embedding_dim,))
                else:
                    word_lower = word.lower()
                    if word_lower in word2vec_model.wv:
                        pretrained_embeddings[idx] = word2vec_model.wv[word_lower]
                        words_found += 1
                    else:
                        pretrained_embeddings[idx] = np.random.normal(scale=0.1, size=(embedding_dim,))
            
            print(f"✅ Word2Vec chargé: {words_found}/{len(vocab)} mots trouvés")
            
        except Exception as e:
            print(f"⚠️ Erreur Word2Vec: {e}")
            return
    
    # 3. Charger le checkpoint du modèle
    print(f"🤖 Chargement du modèle: {model_checkpoint_path}")
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    
    return {
        'vocab': vocab,
        'char_vocab': char_vocab,
        'tag_to_idx': tag_to_idx,
        'idx_to_tag': idx_to_tag,
        'pretrained_embeddings': pretrained_embeddings,
        'checkpoint': checkpoint,
        'device': device
    }