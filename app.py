# streamlit_app.py
import streamlit as st
import torch
import pickle
import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import plotly.express as px
import plotly.graph_objects as go
import time
import json

# Import de votre modèle (ajustez le chemin selon votre structure)
import sys
sys.path.append('..')  # Pour importer depuis le dossier parent
from streamlit_utils import load_all_components

from models.models import CombinatorialNER  # Ajustez selon votre structure

# ============================================
# CONFIGURATION
# ============================================

st.set_page_config(
    page_title="BioNER - Biomedical NER",
    page_icon="🧬",
    layout="wide"
)

# ============================================
# CSS STYLING
# ============================================

st.markdown("""
<style>
    .main-header {
        color: #1E90FF;
        text-align: center;
        padding: 20px;
    }
    .entity-badge {
        display: inline-block;
        padding: 2px 8px;
        margin: 1px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.9em;
    }
    .results-box {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #1E90FF;
    }
    .tab-content {
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# COULEURS DES ENTITÉS
# ============================================

# Entités JNLPBA (11 classes + PAD)
ENTITY_COLORS_JNLPBA = {
    'B-DNA': '#FF6B6B', 'I-DNA': '#FF8E8E',
    'B-RNA': '#4ECDC4', 'I-RNA': '#7FDFD9',
    'B-protein': '#45B7D1', 'I-protein': '#7ACFE5',
    'B-cell_type': '#96CEB4', 'I-cell_type': '#B8E0CD',
    'B-cell_line': "#6D664F", 'I-cell_line': "#C39A12",
    'O': 'transparent',
    '<PAD>': 'transparent'
}

ENTITY_NAMES_JNLPBA = {
    'B-DNA': 'DNA', 'I-DNA': 'DNA',
    'B-RNA': 'RNA', 'I-RNA': 'RNA',
    'B-protein': 'Protein', 'I-protein': 'Protein',
    'B-cell_type': 'Cell Type', 'I-cell_type': 'Cell Type',
    'B-cell_line': 'Cell Line', 'I-cell_line': 'Cell Line',
    'O': 'Other',
    '<PAD>': 'Padding'
}

# Entités NCBI (4 classes : B-Disease, I-Disease, O, <PAD>)
ENTITY_COLORS_NCBI = {
    'B-Disease': '#FF6B6B', 
    'I-Disease': '#FF8E8E',
    'O': 'transparent',
    '<PAD>': 'transparent'
}

ENTITY_NAMES_NCBI = {
    'B-Disease': 'Disease', 
    'I-Disease': 'Disease',
    'O': 'Other',
    '<PAD>': 'Padding'
}

# ============================================
# CLASSES UTILITAIRES
# ============================================

class StreamlitNERPredictor:
    def __init__(self, components: Dict, dataset_name: str = 'JNLPBA',
                 use_char_cnn=True, use_char_lstm=True,
                 use_attention=True, use_fc_fusion=True):
        """Initialise le prédicteur avec tous les composants chargés"""
        self.vocab = components['vocab']
        self.char_vocab = components['char_vocab']
        self.tag_to_idx = components['tag_to_idx']
        self.idx_to_tag = components['idx_to_tag']
        self.pretrained_embeddings = components['pretrained_embeddings']
        self.checkpoint = components['checkpoint']
        self.device = components['device']
        self.dataset_name = dataset_name
        
        # Vérifier la taille des vocabulaires
        print(f"📊 Taille vocab: {len(self.vocab)}, char vocab: {len(self.char_vocab)}, tags: {len(self.tag_to_idx)}")
        
        # Configuration selon le dataset
        if dataset_name == 'JNLPBA':
            lstm_hidden_dim = 256
            # Vérification pour JNLPBA
            expected_tags = 12  # 11 tags + PAD
            if len(self.tag_to_idx) != expected_tags:
                print(f"⚠️ Attention: JNLPBA a {len(self.tag_to_idx)} tags au lieu de {expected_tags}")
        else:  # NCBI
            lstm_hidden_dim = 128
            # Vérification pour NCBI
            expected_tags = 4  # B-Disease, I-Disease, O, <PAD>
            if len(self.tag_to_idx) != expected_tags:
                print(f"⚠️ Attention: NCBI a {len(self.tag_to_idx)} tags au lieu de {expected_tags}")
        
        # Récupérer les paramètres du checkpoint
        checkpoint_data = self.checkpoint
        epoch = checkpoint_data.get('epoch', 0)
        best_f1 = checkpoint_data.get('best_f1', 0.0)
        
        print(f"📦 Checkpoint chargé: dataset={dataset_name}, epoch={epoch}, best_f1={best_f1:.4f}")
        print(f"📊 Classes disponibles: {list(self.idx_to_tag.values())}")
        
        # Créer le modèle avec les mêmes paramètres qu'à l'entraînement
        self.model = CombinatorialNER(
            vocab_size=len(self.vocab),
            char_vocab_size=len(self.char_vocab),
            tag_to_idx=self.tag_to_idx,
            dataset=dataset_name,
            use_char_cnn=use_char_cnn,
            use_char_lstm=use_char_lstm,
            use_attention=use_attention,
            use_fc_fusion=use_fc_fusion,
            pretrained_embeddings=self.pretrained_embeddings,
            word_embed_dim=200,
            lstm_hidden_dim=lstm_hidden_dim,
            dropout=0.5,
            use_lstm=True
        ).to(self.device)
        
        # Charger les poids
        try:
            if 'model_state_dict' in checkpoint_data:
                self.model.load_state_dict(checkpoint_data['model_state_dict'])
                print("✅ Chargé depuis 'model_state_dict'")
            else:
                self.model.load_state_dict(checkpoint_data)
                print("✅ Chargé depuis le checkpoint direct")
            
            print(f"✅ Poids du modèle {dataset_name} chargés avec succès")
            
            # Vérifier les paramètres chargés
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"📊 Paramètres totaux: {total_params:,}")
            
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement: {e}")
            import traceback
            traceback.print_exc()
            
            # Chargement partiel
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_data.items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)
            print(f"✅ Chargement partiel réussi: {len(pretrained_dict)}/{len(checkpoint_data)} paramètres")
        
        self.model.eval()
        print(f"✅ Modèle {self.dataset_name} prêt sur {self.device}")
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenisation simple du texte"""
        # Tokenisation adaptée au texte biomédical
        tokens = re.findall(r'\b\w+(?:-\w+)*\b|[^\w\s]', text)
        return tokens
    
    def preprocess_tokens(self, tokens: List[str], max_seq_len: int = 100, max_char_len: int = 20):
        """Préparation des tokens pour le modèle"""
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        
        seq_len = len(tokens)
        
        # IDs des mots
        word_ids = []
        UNK_WORD = self.vocab.get('<UNK>', 1)
        PAD_WORD = self.vocab.get('<PAD>', 0)
        
        for token in tokens:
            if token.isdigit():
                token_id = self.vocab.get('<NUM>', UNK_WORD)
            else:
                token_lower = token.lower()
                token_id = self.vocab.get(token_lower, UNK_WORD)
            word_ids.append(token_id)
        
        # Padding pour les mots
        word_ids += [PAD_WORD] * (max_seq_len - seq_len)
        
        # Séquences de caractères
        char_seqs = []
        UNK_CHAR = self.char_vocab.get('<UNK>', 1)
        PAD_CHAR = self.char_vocab.get('<PAD>', 0)
        
        for token in tokens:
            chars = [self.char_vocab.get(c, UNK_CHAR) for c in token[:max_char_len]]
            chars += [PAD_CHAR] * (max_char_len - len(chars))
            char_seqs.append(chars)
        
        # Padding pour les caractères
        char_seqs += [[PAD_CHAR] * max_char_len] * (max_seq_len - seq_len)
        
        return tokens, word_ids, char_seqs, seq_len
    
    def predict(self, text: str):
        """Prédiction principale - ADAPTÉ À VOTRE IMPLÉMENTATION"""
        # Tokenisation
        tokens = self.tokenize_text(text)
        
        if not tokens:
            return []
        
        # Préparation
        tokens, word_ids, char_seqs, seq_len = self.preprocess_tokens(tokens)
        
        # Conversion en tensors
        word_tensor = torch.tensor([word_ids], dtype=torch.long).to(self.device)
        char_tensor = torch.tensor([char_seqs], dtype=torch.long).to(self.device)
        
        # Créer le masque (True pour les tokens réels, False pour padding)
        mask = torch.ones((1, 100), dtype=torch.bool).to(self.device)
        mask[:, seq_len:] = False
        
        # Prédiction - adaptation selon votre code
        with torch.no_grad():
            try:
                # Appel direct au modèle comme dans votre code
                predictions = self.model(word_tensor, char_tensor, mask=mask)
                
                # Votre code retourne: predictions[0][:seq_len]
                if isinstance(predictions, list) and len(predictions) > 0:
                    predicted_ids = predictions[0][:seq_len]
                elif isinstance(predictions, tuple) and len(predictions) > 0:
                    predicted_ids = predictions[0][:seq_len]
                else:
                    # Fallback: argmax sur les émissions
                    print("⚠️ Utilisation du fallback (sans CRF)")
                    emissions = self.get_emissions(word_tensor, char_tensor, mask)
                    predicted_ids = torch.argmax(emissions, dim=2)[0][:seq_len].cpu().numpy()
            
            except Exception as e:
                print(f"⚠️ Erreur prédiction: {e}, utilisation du fallback")
                emissions = self.get_emissions(word_tensor, char_tensor, mask)
                predicted_ids = torch.argmax(emissions, dim=2)[0][:seq_len].cpu().numpy()
        
        # Conversion en tags
        pred_tags = []
        for idx in predicted_ids:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            tag = self.idx_to_tag.get(idx, 'O')
            pred_tags.append(tag)
        
        return list(zip(tokens, pred_tags))
    
    def get_emissions(self, word_tensor, char_tensor, mask):
        """Récupère les émissions brutes (sans CRF) pour le fallback"""
        # Forward pass manuel
        word_emb = self.model.word_embedding(word_tensor)
        
        char_embs = []
        if hasattr(self.model, 'use_char_cnn') and self.model.use_char_cnn and hasattr(self.model, 'char_cnn'):
            char_embs.append(self.model.char_cnn(char_tensor))
        if hasattr(self.model, 'use_char_lstm') and self.model.use_char_lstm and hasattr(self.model, 'char_lstm'):
            char_embs.append(self.model.char_lstm(char_tensor))
        
        if char_embs:
            combined = torch.cat([word_emb] + char_embs, dim=-1)
        else:
            combined = word_emb
        
        if hasattr(self.model, 'use_fc_fusion') and self.model.use_fc_fusion and hasattr(self.model, 'fusion'):
            combined = self.model.fusion(combined)
        
        if hasattr(self.model, 'context_lstm') and self.model.context_lstm is not None:
            lstm_out, _ = self.model.context_lstm(combined)
            if hasattr(self.model, 'attention_layer') and self.model.attention_layer is not None:
                lstm_out = self.model.attention_layer(lstm_out, mask)
        else:
            lstm_out = combined
        
        emissions = self.model.emission(lstm_out)
        
        return emissions
    
    def extract_entities(self, predictions: List[Tuple[str, str]]):
        """Extraction des entités des prédictions"""
        entities = []
        current_entity = None
        entity_tokens = []
        entity_type = None
        entity_start_idx = 0
        
        for idx, (token, tag) in enumerate(predictions):
            if tag.startswith('B-'):
                # Sauvegarder l'entité précédente
                if current_entity:
                    entities.append({
                        'text': ' '.join(entity_tokens),
                        'type': entity_type[2:],
                        'tag': entity_type,
                        'tokens': entity_tokens.copy(),
                        'start_position': entity_start_idx,
                        'end_position': idx - 1
                    })
                
                # Nouvelle entité
                current_entity = tag[2:]
                entity_type = tag
                entity_tokens = [token]
                entity_start_idx = idx
                
            elif tag.startswith('I-'):
                if current_entity == tag[2:]:
                    entity_tokens.append(token)
                else:
                    # I- sans B- précédent (traitement comme B-)
                    if current_entity:
                        entities.append({
                            'text': ' '.join(entity_tokens),
                            'type': entity_type[2:],
                            'tag': entity_type,
                            'tokens': entity_tokens.copy(),
                            'start_position': entity_start_idx,
                            'end_position': idx - 1
                        })
                    
                    current_entity = tag[2:]
                    entity_type = 'B-' + tag[2:]  # Convertir en B-
                    entity_tokens = [token]
                    entity_start_idx = idx
            
            else:  # 'O' ou autre
                if current_entity:
                    entities.append({
                        'text': ' '.join(entity_tokens),
                        'type': entity_type[2:],
                        'tag': entity_type,
                        'tokens': entity_tokens.copy(),
                        'start_position': entity_start_idx,
                        'end_position': idx - 1
                    })
                    current_entity = None
                    entity_tokens = []
                    entity_start_idx = 0
        
        # Dernière entité
        if current_entity:
            entities.append({
                'text': ' '.join(entity_tokens),
                'type': entity_type[2:],
                'tag': entity_type,
                'tokens': entity_tokens.copy(),
                'start_position': entity_start_idx,
                'end_position': len(predictions) - 1
            })
        
        return entities

# ============================================
# FONCTIONS UTILITAIRES
# ============================================

@st.cache_resource
def load_jnlpba_components():
    """Charge les composants pour JNLPBA (entités biomédicales)"""
    try:
        # Chemins pour JNLPBA
        model_path = "./checkpoints/JNLPBA/WE/best_model.pt"
        vocab_dir = "./vocab/jnlpba"
        word2vec_path = "./word2Vecembeddings/jnlpba_word2vec"
        
        # Vérifier les fichiers
        if not os.path.exists(model_path):
            st.error(f"❌ Modèle JNLPBA non trouvé: {model_path}")
            return None
        
        if not os.path.exists(vocab_dir):
            st.error(f"❌ Vocabulaire JNLPBA non trouvé: {vocab_dir}")
            return None
        
        st.info(f"📂 Chargement JNLPBA depuis: {model_path}")
        
        # Charger les composants
        components = load_all_components(model_path, vocab_dir, word2vec_path)
        components['checkpoint_path'] = model_path
        
        # Afficher des informations de débogage
        if 'tag_to_idx' in components:
            st.info(f"📊 JNLPBA - Nombre de tags: {len(components['tag_to_idx'])}")
            st.info(f"📊 JNLPBA - Tags: {list(components.get('idx_to_tag', {}).values())}")
        
        # Créer le prédicteur avec les bons paramètres
        predictor = StreamlitNERPredictor(
            components, 
            dataset_name='JNLPBA',
            use_char_cnn=False, 
            use_char_lstm=False,
            use_attention=False, 
            use_fc_fusion=False  # IMPORTANT: False pour JNLPBA selon votre code
        )
        
        return predictor
        
    except Exception as e:
        st.error(f"❌ Erreur JNLPBA: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

@st.cache_resource
def load_ncbi_components():
    """Charge les composants pour NCBI (maladies)"""
    try:
        # Chemins pour NCBI - CORRIGÉ selon votre structure
        model_path = "./checkpoints/NCBI/WE_char_bilstm_cnn_attention/best_model.pt"
        vocab_dir = "./vocab/ncbi"  # Dossier du vocabulaire NCBI
        word2vec_path = "./word2Vecembeddings/ncbi.model"  # Embeddings NCBI
        
        # Vérifier les fichiers
        if not os.path.exists(model_path):
            st.error(f"❌ Modèle NCBI non trouvé: {model_path}")
            st.error(f"Recherche à: {os.path.abspath(model_path)}")
            return None
        
        if not os.path.exists(vocab_dir):
            st.error(f"❌ Vocabulaire NCBI non trouvé: {vocab_dir}")
            st.error(f"Recherche à: {os.path.abspath(vocab_dir)}")
            return None
        
        st.info(f"📂 Chargement NCBI depuis: {model_path}")
        
        # Charger les composants
        components = load_all_components(model_path, vocab_dir, word2vec_path)
        components['checkpoint_path'] = model_path
        
        # Afficher des informations de débogage
        if 'tag_to_idx' in components:
            st.info(f"📊 NCBI - Nombre de tags: {len(components['tag_to_idx'])}")
            st.info(f"📊 NCBI - Tags: {list(components.get('idx_to_tag', {}).values())}")
        
        predictor = StreamlitNERPredictor(
            components, 
            dataset_name='NCBI',
            use_char_cnn=True, 
            use_char_lstm=True,
            use_attention=True, 
            use_fc_fusion=False  
        )
        
        return predictor
        
    except Exception as e:
        st.error(f"❌ Erreur NCBI: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def highlight_text(text: str, predictions: List[Tuple[str, str]], dataset: str = 'JNLPBA'):
    """Surligne le texte avec les entités"""
    entity_colors = ENTITY_COLORS_JNLPBA if dataset == 'JNLPBA' else ENTITY_COLORS_NCBI
    entity_names = ENTITY_NAMES_JNLPBA if dataset == 'JNLPBA' else ENTITY_NAMES_NCBI
    
    highlighted = []
    for token, tag in predictions:
        if tag != 'O' and tag != '<PAD>':
            color = entity_colors.get(tag, '#CCCCCC')
            entity_name = entity_names.get(tag, tag[2:] if tag.startswith(('B-', 'I-')) else tag)
            highlighted.append(f'<span class="entity-badge" style="background-color: {color};" title="{entity_name}">{token}</span>')
        else:
            highlighted.append(token)
    
    return ' '.join(highlighted)

def create_entity_legend(dataset: str = 'JNLPBA'):
    """Crée la légende des entités selon le dataset"""
    if dataset == 'JNLPBA':
        entity_colors = ENTITY_COLORS_JNLPBA
        entity_names = ENTITY_NAMES_JNLPBA
        title = "🎨 Types d'Entités Biomédicales"
    else:  # NCBI
        entity_colors = ENTITY_COLORS_NCBI
        entity_names = ENTITY_NAMES_NCBI
        title = "🎨 Types d'Entités (NCBI)"
    
    st.markdown(f"### {title}")
    
    entity_items = []
    
    for tag, color in entity_colors.items():
        if tag not in ['O', '<PAD>'] and tag.startswith('B-'):
            entity_name = entity_names.get(tag, tag[2:])
            entity_items.append((entity_name, color))
    
    # Afficher dans des colonnes
    if entity_items:
        cols = st.columns(min(4, len(entity_items)))
        items_per_col = len(entity_items) // len(cols) + 1
        
        for i, col in enumerate(cols):
            start_idx = i * items_per_col
            end_idx = min((i + 1) * items_per_col, len(entity_items))
            
            with col:
                for entity_name, color in entity_items[start_idx:end_idx]:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <div style="width: 15px; height: 15px; background-color: {color}; margin-right: 8px; border-radius: 3px;"></div>
                        <span>{entity_name}</span>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Aucun type d'entité configuré")

def display_debug_info(predictions, entities, dataset):
    """Affiche des informations de débogage"""
    with st.expander("🔍 Informations de débogage"):
        st.write("**Predictions brutes:**")
        for token, tag in predictions:
            st.write(f"- '{token}' → {tag}")
        
        st.write(f"\n**Nombre d'entités extraites:** {len(entities)}")
        st.write(f"**Dataset:** {dataset}")
        
        if entities:
            st.write("\n**Entités détaillées:**")
            for i, entity in enumerate(entities, 1):
                st.write(f"{i}. Texte: '{entity['text']}', Type: {entity['type']}, Tags: {entity['tokens']}")

# ============================================
# PAGES DE L'APPLICATION
# ============================================

def biomedical_ner_page():
    """Page pour les entités biomédicales (JNLPBA)"""
    st.markdown('<h1 class="main-header">🧬 Biomedical Named Entity Recognition</h1>', unsafe_allow_html=True)
    st.markdown("Extract biomedical entities (DNA, RNA, proteins, cells) from text using deep learning")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        # Charger le modèle JNLPBA
        if 'predictor_jnlpba' not in st.session_state:
            with st.spinner("Chargement du modèle JNLPBA..."):
                predictor = load_jnlpba_components()
                if predictor:
                    st.session_state.predictor_jnlpba = predictor
                    st.success("✅ Modèle JNLPBA chargé!")
                    
                    # Afficher les classes
                    if hasattr(predictor, 'idx_to_tag'):
                        tags = list(predictor.idx_to_tag.values())
                        st.info(f"**Classes JNLPBA:** {len(tags)} tags")
                        for tag in tags:
                            if tag != '<PAD>':
                                st.write(f"- {tag}")
                else:
                    st.error("❌ Échec du chargement")
                    st.stop()
        
        predictor = st.session_state.predictor_jnlpba
        
        st.markdown("---")
        st.markdown("### 📊 Informations")
        st.markdown(f"""
        - **Dataset:** {predictor.dataset_name}
        - **Vocabulaire:** {len(predictor.vocab)} mots
        - **Classes d'entités:** {len([t for t in predictor.idx_to_tag.values() if t not in ['O', '<PAD>']])}
        - **Tags totaux:** {len(predictor.tag_to_idx)}
        - **Device:** {predictor.device}
        """)
        
        # Option de débogage - CORRIGÉ: utilisation directe du widget
        st.markdown("---")
        debug_jnlpba = st.checkbox("Afficher les infos de débogage", key="debug_jnlpba_checkbox")
    
    # Légende des entités
    create_entity_legend('JNLPBA')
    
    st.markdown("---")
    
    # Zone de texte
    st.markdown("### 📝 Entrez votre texte biomédical")
    
    # Exemples pour JNLPBA
    examples = {
        "Génétique": (
            "Mutations in the TP53 gene are frequently observed in human cancers and lead to loss of p53 protein "
            "tumor suppressor activity. Overexpression of MDM2 results in increased degradation of p53, while "
            "alterations in BRCA1 and BRCA2 genes impair DNA double-strand break repair through homologous recombination. "
            "Recent studies also indicate that ATM and ATR kinases phosphorylate p53 in response to DNA damage."
        ),
        "Immunologie": (
            "Activation of T lymphocytes requires signaling through the T cell receptor complex and costimulatory "
            "molecules such as CD28. IL-2 gene transcription is regulated by NF-kappa B, AP-1, and NFAT transcription factors. "
            "Inhibition of JAK3 signaling suppresses STAT5 phosphorylation and reduces IL-2 mRNA expression in activated T cells."
        ),
        "Cellulaire": (
            "HeLa cells and HEK293 cell lines are widely used to study transcriptional regulation and protein-protein interactions. "
            "Jurkat T cells exhibit strong activation of MAPK and ERK signaling pathways following stimulation with phorbol esters. "
            "Primary fibroblasts show increased expression of collagen genes during wound healing."
        )
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧬 Exemple Génétique", use_container_width=True, key="ex1_jnlpba"):
            st.session_state.example_text_jnlpba = examples["Génétique"]
    with col2:
        if st.button("🩸 Exemple Immunologie", use_container_width=True, key="ex2_jnlpba"):
            st.session_state.example_text_jnlpba = examples["Immunologie"]
    with col3:
        if st.button("🔬 Exemple Cellulaire", use_container_width=True, key="ex3_jnlpba"):
            st.session_state.example_text_jnlpba = examples["Cellulaire"]
    
    # Zone de texte
    text_input = st.text_area(
        "**Texte à analyser:**",
        value=st.session_state.get('example_text_jnlpba', ''),
        height=200,
        placeholder="Collez votre texte biomédical ici...",
        key="text_area_jnlpba"
    )
    
    # Bouton de prédiction
    col1, col2 = st.columns([3, 1])
    with col2:
        analyze = st.button("🔍 Analyser le texte", type="primary", use_container_width=True, key="analyze_jnlpba")
    
    if analyze:
        if not text_input.strip():
            st.error("❌ Veuillez entrer du texte.")
        else:
            with st.spinner("Analyse en cours..."):
                start_time = time.time()
                
                try:
                    # Prédiction
                    predictions = predictor.predict(text_input)
                    entities = predictor.extract_entities(predictions)
                    
                    processing_time = time.time() - start_time
                    
                    # Stocker les résultats
                    st.session_state.last_results_jnlpba = {
                        'predictions': predictions,
                        'entities': entities,
                        'text': text_input,
                        'processing_time': processing_time,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'dataset': 'JNLPBA'
                    }
                    
                    st.success(f"✅ {len(entities)} entités trouvées en {processing_time:.2f} secondes!")
                    
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Afficher les résultats
    if 'last_results_jnlpba' in st.session_state:
        results = st.session_state.last_results_jnlpba
        
        st.markdown("---")
        st.markdown("### 📊 Résultats")
        
        # Métriques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Entités trouvées", len(results['entities']))
        with col2:
            st.metric("Temps d'analyse", f"{results['processing_time']:.2f}s")
        with col3:
            unique_types = len(set([e['type'] for e in results['entities']]))
            st.metric("Types d'entités", unique_types)
        
        # Onglets
        tab_names = ["📄 Texte annoté", "📊 Liste des entités", "📈 Statistiques"]
        
        # Ajouter l'onglet débogage si l'option est activée
        if debug_jnlpba:
            tab_names.append("🔍 Détails")
        
        tabs = st.tabs(tab_names)
        
        with tabs[0]:  # Texte annoté
            st.markdown("#### Texte avec entités surlignées")
            highlighted = highlight_text(results['text'], results['predictions'], 'JNLPBA')
            st.markdown(f'<div class="results-box">{highlighted}</div>', unsafe_allow_html=True)
        
        with tabs[1]:  # Liste des entités
            if results['entities']:
                df_data = []
                for entity in results['entities']:
                    entity_name = ENTITY_NAMES_JNLPBA.get(entity['tag'], entity['type'])
                    df_data.append({
                        'Entité': entity['text'],
                        'Type': entity_name,
                        'Tag': entity['tag'],
                        'Tokens': len(entity['tokens']),
                        'Position': f"{entity['start_position']}-{entity['end_position']}"
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ Aucune entité trouvée.")
        
        with tabs[2]:  # Statistiques
            if results['entities']:
                # Distribution par type
                type_counts = {}
                for entity in results['entities']:
                    entity_type = ENTITY_NAMES_JNLPBA.get(entity['tag'], entity['type'])
                    type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
                
                # Graphique
                if type_counts:
                    fig = px.bar(
                        x=list(type_counts.keys()),
                        y=list(type_counts.values()),
                        title="Distribution des types d'entités",
                        labels={'x': 'Type', 'y': 'Nombre'},
                        color=list(type_counts.keys()),
                        color_discrete_map={
                            'DNA': '#FF6B6B',
                            'RNA': '#4ECDC4',
                            'Protein': '#45B7D1',
                            'Cell Type': '#96CEB4',
                            'Cell Line': '#6D664F'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Longueur moyenne des entités
                avg_length = np.mean([len(e['tokens']) for e in results['entities']])
                st.metric("Longueur moyenne", f"{avg_length:.1f} tokens")
        
        # Onglet débogage (si activé)
        if debug_jnlpba and len(tabs) > 3:
            with tabs[3]:  # Détails débogage
                display_debug_info(results['predictions'], results['entities'], 'JNLPBA')
        
        # Export
        st.markdown("---")
        st.markdown("### 💾 Exporter les résultats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export JSON
            export_data = {
                'text': results['text'],
                'entities': results['entities'],
                'timestamp': results['timestamp'],
                'processing_time': results['processing_time'],
                'dataset': results['dataset']
            }
            
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Télécharger JSON",
                data=json_str,
                file_name="bio_ner_results_jnlpba.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Export CSV
            if results['entities']:
                df_data = []
                for entity in results['entities']:
                    df_data.append({
                        'entity': entity['text'],
                        'type': ENTITY_NAMES_JNLPBA.get(entity['tag'], entity['type']),
                        'tag': entity['tag'],
                        'tokens': ' '.join(entity['tokens'])
                    })
                
                df = pd.DataFrame(df_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Télécharger CSV",
                    data=csv,
                    file_name="bio_ner_entities_jnlpba.csv",
                    mime="text/csv",
                    use_container_width=True
                )

def disease_ner_page():
    """Page pour les entités de maladies (NCBI)"""
    st.markdown('<h1 class="main-header">🩺 Disease Named Entity Recognition</h1>', unsafe_allow_html=True)
    st.markdown("Extract disease entities from biomedical text using deep learning")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        # Charger le modèle NCBI
        if 'predictor_ncbi' not in st.session_state:
            with st.spinner("Chargement du modèle NCBI (maladies)..."):
                predictor = load_ncbi_components()
                if predictor:
                    st.session_state.predictor_ncbi = predictor
                    st.success("✅ Modèle NCBI chargé!")
                    
                    # Afficher les classes
                    if hasattr(predictor, 'idx_to_tag'):
                        tags = list(predictor.idx_to_tag.values())
                        st.info(f"**Classes NCBI:** {len(tags)} tags")
                        for tag in tags:
                            if tag != '<PAD>':
                                st.write(f"- {tag}")
                else:
                    st.error("❌ Échec du chargement")
                    st.stop()
        
        predictor = st.session_state.predictor_ncbi
        
        st.markdown("---")
        st.markdown("### 📊 Informations")
        st.markdown(f"""
        - **Dataset:** {predictor.dataset_name} (Diseases)
        - **Vocabulaire:** {len(predictor.vocab)} mots
        - **Classes d'entités:** {len([t for t in predictor.idx_to_tag.values() if t not in ['O', '<PAD>']])}
        - **Tags totaux:** {len(predictor.tag_to_idx)}
        - **Device:** {predictor.device}
        """)
        
        # Option de débogage - CORRIGÉ
        st.markdown("---")
        debug_ncbi = st.checkbox("Afficher les infos de débogage", key="debug_ncbi_checkbox")
    
    # Légende des entités
    create_entity_legend('NCBI')
    
    st.markdown("---")
    
    # Zone de texte
    st.markdown("### 📝 Entrez votre texte biomédical")
    
    # Exemples pour NCBI (maladies) - adaptés aux 4 classes
    examples = {
        "Cancer": (
            "The hereditary breast and ovarian cancer syndrome is associated with a high frequency of BRCA1 mutations. "
            "Patients with BRCA1 mutation show increased risk of developing breast cancer and ovarian cancer. "
            "TP53 mutations are also frequently observed in various human cancers."
        ),
        "Maladies Génétiques": (
            "Cystic fibrosis is caused by mutations in the CFTR gene and affects the lungs and digestive system. "
            "Huntington's disease is a neurodegenerative disorder caused by a CAG repeat expansion in the HTT gene. "
            "Familial hypercholesterolemia results from mutations in the LDLR gene."
        ),
        "Maladies Infectieuses": (
            "The COVID-19 pandemic caused by SARS-CoV-2 has affected millions worldwide. "
            "HIV infection leads to acquired immunodeficiency syndrome (AIDS) by destroying CD4+ T cells. "
            "Tuberculosis remains a major global health problem, especially multidrug-resistant tuberculosis."
        ),
        "Test Simple": (
            "Breast cancer and ovarian cancer are common diseases. "
            "Diabetes is a chronic condition affecting millions."
        )
    }
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🎗️ Exemple Cancer", use_container_width=True, key="ex4_ncbi"):
            st.session_state.example_text_ncbi = examples["Cancer"]
    with col2:
        if st.button("🧬 Exemple Génétique", use_container_width=True, key="ex5_ncbi"):
            st.session_state.example_text_ncbi = examples["Maladies Génétiques"]
    with col3:
        if st.button("🦠 Exemple Infectieuses", use_container_width=True, key="ex6_ncbi"):
            st.session_state.example_text_ncbi = examples["Maladies Infectieuses"]
    with col4:
        if st.button("🧪 Test Simple", use_container_width=True, key="ex7_ncbi"):
            st.session_state.example_text_ncbi = examples["Test Simple"]
    
    # Zone de texte
    text_input = st.text_area(
        "**Texte à analyser:**",
        value=st.session_state.get('example_text_ncbi', ''),
        height=200,
        placeholder="Collez votre texte biomédical ici...",
        key="text_area_ncbi"
    )
    
    # Bouton de prédiction
    col1, col2 = st.columns([3, 1])
    with col2:
        analyze = st.button("🔍 Analyser le texte", type="primary", use_container_width=True, key="analyze_ncbi")
    
    if analyze:
        if not text_input.strip():
            st.error("❌ Veuillez entrer du texte.")
        else:
            with st.spinner("Analyse en cours..."):
                start_time = time.time()
                
                try:
                    # Prédiction
                    predictions = predictor.predict(text_input)
                    entities = predictor.extract_entities(predictions)
                    
                    processing_time = time.time() - start_time
                    
                    # Stocker les résultats
                    st.session_state.last_results_ncbi = {
                        'predictions': predictions,
                        'entities': entities,
                        'text': text_input,
                        'processing_time': processing_time,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'dataset': 'NCBI'
                    }
                    
                    st.success(f"✅ {len(entities)} maladies trouvées en {processing_time:.2f} secondes!")
                    
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Afficher les résultats
    if 'last_results_ncbi' in st.session_state:
        results = st.session_state.last_results_ncbi
        
        st.markdown("---")
        st.markdown("### 📊 Résultats")
        
        # Métriques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Maladies trouvées", len(results['entities']))
        with col2:
            st.metric("Temps d'analyse", f"{results['processing_time']:.2f}s")
        with col3:
            if results['entities']:
                avg_length = np.mean([len(e['tokens']) for e in results['entities']])
                st.metric("Longueur moyenne", f"{avg_length:.1f} mots")
            else:
                st.metric("Longueur moyenne", "0")
        
        # Onglets
        tab_names = ["📄 Texte annoté", "📊 Liste des maladies", "📈 Statistiques"]
        
        # Ajouter l'onglet débogage si l'option est activée
        if debug_ncbi:
            tab_names.append("🔍 Détails")
        
        tabs = st.tabs(tab_names)
        
        with tabs[0]:  # Texte annoté
            st.markdown("#### Texte avec maladies surlignées")
            highlighted = highlight_text(results['text'], results['predictions'], 'NCBI')
            st.markdown(f'<div class="results-box">{highlighted}</div>', unsafe_allow_html=True)
        
        with tabs[1]:  # Liste des maladies
            if results['entities']:
                df_data = []
                for entity in results['entities']:
                    entity_name = ENTITY_NAMES_NCBI.get(entity['tag'], entity['type'])
                    df_data.append({
                        'Maladie': entity['text'],
                        'Type': entity_name,
                        'Tag': entity['tag'],
                        'Mots': len(entity['tokens']),
                        'Position': f"{entity['start_position']}-{entity['end_position']}"
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Afficher un résumé
                st.markdown("#### 📋 Résumé des maladies trouvées:")
                for i, entity in enumerate(results['entities'], 1):
                    st.write(f"{i}. **{entity['text']}** ({len(entity['tokens'])} mots, tag: {entity['tag']})")
            else:
                st.info("ℹ️ Aucune maladie trouvée.")
        
        with tabs[2]:  # Statistiques
            if results['entities']:
                # Distribution par longueur
                lengths = [len(e['tokens']) for e in results['entities']]
                
                if lengths:
                    fig = px.histogram(
                        x=lengths,
                        title="Distribution des longueurs des maladies",
                        labels={'x': 'Nombre de mots', 'y': 'Fréquence'},
                        nbins=10
                    )
                    fig.update_layout(
                        xaxis_title="Nombre de mots par maladie",
                        yaxis_title="Nombre de maladies"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistiques descriptives
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Moyenne", f"{np.mean(lengths):.1f}")
                    with col2:
                        st.metric("Médiane", f"{np.median(lengths):.1f}")
                    with col3:
                        st.metric("Min", f"{min(lengths)}")
                    with col4:
                        st.metric("Max", f"{max(lengths)}")
            else:
                st.info("ℹ️ Aucune statistique disponible (pas de maladies trouvées)")
        
        # Onglet débogage (si activé)
        if debug_ncbi and len(tabs) > 3:
            with tabs[3]:  # Détails débogage
                display_debug_info(results['predictions'], results['entities'], 'NCBI')
        
        # Export
        st.markdown("---")
        st.markdown("### 💾 Exporter les résultats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export JSON
            export_data = {
                'text': results['text'],
                'entities': results['entities'],
                'timestamp': results['timestamp'],
                'processing_time': results['processing_time'],
                'dataset': results['dataset']
            }
            
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Télécharger JSON",
                data=json_str,
                file_name="disease_ner_results_ncbi.json",
                mime="application/json",
                use_container_width=True,
                key="download_json_ncbi"
            )
        
        with col2:
            # Export CSV
            if results['entities']:
                df_data = []
                for entity in results['entities']:
                    df_data.append({
                        'disease': entity['text'],
                        'type': ENTITY_NAMES_NCBI.get(entity['tag'], entity['type']),
                        'tag': entity['tag'],
                        'token_count': len(entity['tokens']),
                        'tokens': ' '.join(entity['tokens'])
                    })
                
                df = pd.DataFrame(df_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Télécharger CSV",
                    data=csv,
                    file_name="disease_entities_ncbi.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_csv_ncbi"
                )

def about_page():
    """Page À propos"""
    st.markdown('<h1 class="main-header">ℹ️ À propos</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Application de Reconnaissance d'Entités Nommées Biomédicales
    
    Cette application permet d'extraire des entités nommées à partir de textes biomédicaux en utilisant des modèles de deep learning.
    
    #### Fonctionnalités :
    
    **1. Page Biomedical NER (JNLPBA)**
    - Extraction d'entités biomédicales générales
    - 5 types d'entités : ADN, ARN, protéines, types de cellules, lignées cellulaires
    - 11 tags BIO (B-, I- pour chaque type + O)
    - Modèle entraîné sur le dataset JNLPBA
    
    **2. Page Disease NER (NCBI)**
    - Extraction spécifique de maladies
    - 1 type d'entité : Maladie
    - 3 tags : B-Disease, I-Disease, O (plus <PAD>)
    - Modèle entraîné sur le dataset NCBI
    
    #### Modèles utilisés :
    - **Architecture** : BiLSTM avec attention et CNN de caractères
    - **Embeddings** : Word2Vec pré-entraînés spécifiques à chaque dataset
    - **CRF** : Conditional Random Fields pour le décodage
    
    #### Statistiques des datasets :
    - **JNLPBA** : 12,664 mots, 85 caractères, 12 classes
    - **NCBI** : 5,747 mots, 86 caractères, 4 classes
    
    #### Technologies :
    - PyTorch pour le deep learning
    - Streamlit pour l'interface
    - Plotly pour la visualisation
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Comparaison des datasets")
        st.markdown("""
        | Feature | JNLPBA | NCBI |
        |---------|--------|------|
        | Type d'entités | 5 | 1 |
        | Tags BIO | 11 | 3 |
        | Vocabulaire | 12,664 | 5,747 |
        | Caractères | 85 | 86 |
        | Entités B | 10.1% | 3.5% |
        | Entités I | 11.6% | 3.9% |
        | Autres (O) | 78.3% | 92.6% |
        """)
    
    with col2:
        st.markdown("#### 🚀 Comment utiliser")
        st.markdown("""
        1. **Choisissez une page** (Biomedical ou Disease)
        2. **Entrez ou collez** votre texte biomédical
        3. **Cliquez** sur "Analyser le texte"
        4. **Visualisez** les résultats dans les onglets
        5. **Exportez** en JSON ou CSV si nécessaire
        
        **Astuces :**
        - Utilisez les boutons d'exemple pour tester rapidement
        - Activez le mode débogage pour voir les détails
        - Vérifiez les classes disponibles dans la sidebar
        """)
    
    st.markdown("---")
    st.markdown("#### 📞 Support")
    st.markdown("""
    Pour toute question ou problème :
    - Vérifiez que les chemins des modèles sont corrects
    - Activez le mode débogage pour plus d'informations
    - Contactez l'administrateur pour les problèmes techniques
    """)

# ============================================
# NAVIGATION PRINCIPALE
# ============================================

def main():
    # Sidebar pour la navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        # Sélection de la page
        page = st.radio(
            "Choisissez une page:",
            ["🏥 Biomedical NER", "🩺 Disease NER", "ℹ️ À propos"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("#### 📁 Chemins configurés")
        st.markdown("""
        **JNLPBA:**
        - Modèle: `./checkpoints/JNLPBA/...`
        - Vocab: `./vocab/jnlpba`
        - Embeddings: `./word2Vecembeddings/jnlpba_word2vec`
        
        **NCBI:**
        - Modèle: `./checkpoints/NCBI/...`
        - Vocab: `./vocab/ncbi`
        - Embeddings: `./word2Vecembeddings/ncbi.model`
        """)
    
    # Afficher la page sélectionnée
    if page == "🏥 Biomedical NER":
        biomedical_ner_page()
    elif page == "🩺 Disease NER":
        disease_ner_page()
    elif page == "ℹ️ À propos":
        about_page()

# ============================================
# SCRIPT PRINCIPAL
# ============================================

if __name__ == "__main__":
    # Initialisation des états de session
    if 'example_text_jnlpba' not in st.session_state:
        st.session_state.example_text_jnlpba = ""
    if 'example_text_ncbi' not in st.session_state:
        st.session_state.example_text_ncbi = ""
    
    main()