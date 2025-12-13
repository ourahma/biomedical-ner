import streamlit as st
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re
import tempfile
from typing import List, Dict, Tuple
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import base64
import time
import sys
import os

# ============================================
# CONFIGURATION DES IMPORTS
# ============================================

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'models'))
sys.path.insert(0, os.path.join(current_dir, 'utils'))

# ============================================
# CONFIGURATION DE STREAMLIT
# ============================================

st.set_page_config(
    page_title="BioNER - Biomedical Named Entity Recognition",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS MODE SOMBRE
# ============================================

st.markdown("""
<style>
    /* Mode sombre */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #60a5fa;
    }
    
    /* Text */
    p, span, div {
        color: #e2e8f0;
    }
    
    /* Text areas et inputs */
    .stTextArea textarea, .stTextInput input {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border-color: #475569 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background-color: #1e293b;
        padding: 10px;
        border-radius: 8px;
    }
    
    /* Dataframe */
    .dataframe {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }
    
    .dataframe th {
        background-color: #334155 !important;
        color: #e2e8f0 !important;
    }
    
    .dataframe td {
        background-color: #1e293b !important;
        color: #cbd5e1 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e293b;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        color: #cbd5e1;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    /* Entity badges */
    .entity-badge {
        display: inline-block;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.9em;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Legend */
    .legend-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 15px 0;
        padding: 15px;
        background-color: #1e293b;
        border-radius: 8px;
        border: 1px solid #475569;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        margin-right: 15px;
    }
    
    .legend-color {
        width: 15px;
        height: 15px;
        border-radius: 3px;
        margin-right: 5px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .legend-text {
        font-size: 0.9em;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# IMPORT DES MODÈLES
# ============================================

@st.cache_resource
def load_model_components():
    """Charge les composants du modèle une seule fois"""
    try:
        from models.bioner_model import BiomedicalNERModel
        
        return {
            'BiomedicalNERModel': BiomedicalNERModel,
            'success': True
        }
    except ImportError as e:
        st.error(f"❌ Erreur d'import: {e}")
        st.error("Assurez-vous que le fichier models/bioner_model.py existe")
        return {'success': False}

# Charger les composants
model_components = load_model_components()

if not model_components['success']:
    st.stop()

BiomedicalNERModel = model_components['BiomedicalNERModel']

# ============================================
# CONFIGURATION DES COULEURS DES ENTITÉS
# ============================================

ENTITY_COLORS = {
    'B-DNA': '#FF6B6B',
    'I-DNA': '#FF8E8E',
    'B-RNA': '#4ECDC4',
    'I-RNA': '#7FDFD9',
    'B-protein': '#45B7D1',
    'I-protein': '#7ACFE5',
    'B-cell_type': '#96CEB4',
    'I-cell_type': '#B8E0CD',
    'B-cell_line': '#FFEAA7',
    'I-cell_line': '#FFF4D1',
    'B-DiseaseClass': '#DDA0DD',
    'I-DiseaseClass': '#E6C3E6',
    'B-SpecificDisease': '#98D8C8',
    'I-SpecificDisease': '#C1E8DD',
    'B-CompositeMention': '#F7DC6F',
    'I-CompositeMention': '#FAE8A6',
    'B-Modifier': '#A29BFE',
    'I-Modifier': '#C7C3FE',
    'O': 'transparent'
}

ENTITY_NAMES = {
    'B-DNA': 'DNA',
    'I-DNA': 'DNA',
    'B-RNA': 'RNA',
    'I-RNA': 'RNA',
    'B-protein': 'Protein',
    'I-protein': 'Protein',
    'B-cell_type': 'Cell Type',
    'I-cell_type': 'Cell Type',
    'B-cell_line': 'Cell Line',
    'I-cell_line': 'Cell Line',
    'B-DiseaseClass': 'Disease Class',
    'I-DiseaseClass': 'Disease Class',
    'B-SpecificDisease': 'Specific Disease',
    'I-SpecificDisease': 'Specific Disease',
    'B-CompositeMention': 'Composite Mention',
    'I-CompositeMention': 'Composite Mention',
    'B-Modifier': 'Modifier',
    'I-Modifier': 'Modifier'
}

# ============================================
# CLASSE PREDICTEUR
# ============================================

class NERPredictor:
    """Class to handle NER predictions"""
    
    def __init__(self, model_path: str, vocabs_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load vocabs
        with open(vocabs_path, 'r', encoding='utf-8') as f:
            vocabs = json.load(f)
        
        self.word_vocab = vocabs['word_vocab']
        self.char_vocab = vocabs['char_vocab']
        self.label_vocab = vocabs['label_vocab']
        self.idx_to_label = {v: k for k, v in self.label_vocab.items()}
        
        # Load model config
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extraire la config
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
        else:
            config = {'word_emb_dim': 200, 'char_emb_dim': 100, 'use_attention': True}
        
        # Créer le modèle
        self.model = BiomedicalNERModel(
            word_vocab_size=len(self.word_vocab),
            char_vocab_size=len(self.char_vocab),
            label_vocab_size=len(self.label_vocab),
            word_emb_dim=config.get('word_emb_dim', 200),
            char_emb_dim=config.get('char_emb_dim', 100),
            use_attention=config.get('use_attention', True)
        ).to(self.device)
        
        # Charger les poids
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    def preprocess_text(self, text: str, max_seq_len: int = 100, max_word_len: int = 20):
        tokens = text.split()
        
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        
        word_ids = []
        char_ids = []
        
        for token in tokens:
            word_lower = token.lower()
            if re.match(r'^\d+$', token):
                word_id = self.word_vocab.get('<NUM>', self.word_vocab['<UNK>'])
            else:
                word_id = self.word_vocab.get(word_lower, self.word_vocab['<UNK>'])
            word_ids.append(word_id)
            
            token_chars = []
            for char in token[:max_word_len]:
                token_chars.append(self.char_vocab.get(char, self.char_vocab['<UNK>']))
            
            while len(token_chars) < max_word_len:
                token_chars.append(self.char_vocab['<PAD>'])
            
            char_ids.append(token_chars)
        
        while len(word_ids) < max_seq_len:
            word_ids.append(self.word_vocab['<PAD>'])
            char_ids.append([self.char_vocab['<PAD>']] * max_word_len)
        
        return tokens, word_ids, char_ids
    
    def predict(self, text: str):
        tokens, word_ids, char_ids = self.preprocess_text(text)
        
        words_tensor = torch.tensor([word_ids], dtype=torch.long).to(self.device)
        chars_tensor = torch.tensor([char_ids], dtype=torch.long).to(self.device)
        
        mask = (words_tensor != self.word_vocab['<PAD>']).float()
        
        with torch.no_grad():
            predictions = self.model.predict(words_tensor, chars_tensor, mask)
        
        pred_labels = [self.idx_to_label[pred.item()] for pred in predictions[0, :len(tokens)]]
        
        return list(zip(tokens, pred_labels))
    
    def extract_entities(self, predictions: List[Tuple[str, str]]):
        entities = []
        current_entity = None
        entity_tokens = []
        entity_type = None
        
        for token, tag in predictions:
            if tag.startswith('B-'):
                if current_entity:
                    entities.append({
                        'text': ' '.join(entity_tokens),
                        'type': entity_type[2:],
                        'start_tag': entity_type,
                        'tokens': entity_tokens.copy()
                    })
                
                current_entity = tag[2:]
                entity_type = tag
                entity_tokens = [token]
                
            elif tag.startswith('I-'):
                if current_entity == tag[2:]:
                    entity_tokens.append(token)
                else:
                    if current_entity:
                        entities.append({
                            'text': ' '.join(entity_tokens),
                            'type': entity_type[2:],
                            'start_tag': entity_type,
                            'tokens': entity_tokens.copy()
                        })
                    
                    current_entity = tag[2:]
                    entity_type = tag
                    entity_tokens = [token]
            
            else:
                if current_entity:
                    entities.append({
                        'text': ' '.join(entity_tokens),
                        'type': entity_type[2:],
                        'start_tag': entity_type,
                        'tokens': entity_tokens.copy()
                    })
                    current_entity = None
                    entity_tokens = []
        
        if current_entity:
            entities.append({
                'text': ' '.join(entity_tokens),
                'type': entity_type[2:],
                'start_tag': entity_type,
                'tokens': entity_tokens.copy()
            })
        
        return entities

# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def create_entity_legend():
    """Crée la légende des entités avec des colonnes de Streamlit"""
    # Créer des colonnes pour la légende
    cols = st.columns(4)  # 4 colonnes
    
    # Grouper les entités par colonne
    entity_items = []
    for entity_tag, color in ENTITY_COLORS.items():
        if entity_tag != 'O' and entity_tag.startswith('B-'):
            entity_name = ENTITY_NAMES.get(entity_tag, entity_tag[2:])
            entity_items.append((entity_name, color))
    
    # Répartir les items dans les colonnes
    items_per_col = len(entity_items) // 4 + 1
    
    for i, col in enumerate(cols):
        start_idx = i * items_per_col
        end_idx = min((i + 1) * items_per_col, len(entity_items))
        
        with col:
            for entity_name, color in entity_items[start_idx:end_idx]:
                # Utiliser markdown avec unsafe_allow_html pour afficher les couleurs
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <div style="width: 15px; height: 15px; background-color: {color}; margin-right: 8px; border-radius: 3px; border: 1px solid rgba(255,255,255,0.1);"></div>
                    <span style="color: #e2e8f0; font-size: 0.9em;">{entity_name}</span>
                </div>
                """, unsafe_allow_html=True)

def highlight_text_with_entities(text: str, entities: List[Dict]):
    """Crée du HTML avec les entités surlignées"""
    # Trouver les positions des entités dans le texte
    for entity in entities:
        entity_text = entity['text']
        start_idx = text.find(entity_text)
        if start_idx != -1:
            entity['start_idx'] = start_idx
            entity['end_idx'] = start_idx + len(entity_text)
    
    # Trier par position
    entities = sorted(entities, key=lambda x: x.get('start_idx', 0))
    
    # Construire le HTML
    highlighted_html = ""
    last_idx = 0
    
    for entity in entities:
        start_idx = entity.get('start_idx', -1)
        end_idx = entity.get('end_idx', -1)
        
        if start_idx != -1 and end_idx != -1:
            # Ajouter le texte avant l'entité
            highlighted_html += text[last_idx:start_idx]
            
            # Ajouter l'entité surlignée
            color = ENTITY_COLORS.get(entity['start_tag'], '#CCCCCC')
            entity_name = ENTITY_NAMES.get(entity['start_tag'], entity['type'])
            text_color = '#000' if color not in ['#FFF4D1', '#FAE8A6', '#FFEAA7'] else '#000'
            highlighted_html += f"""
            <span class="entity-badge" style="background-color: {color}; color: {text_color};" title="{entity_name}">
                {text[start_idx:end_idx]}
            </span>
            """
            
            last_idx = end_idx
    
    # Ajouter le texte restant
    highlighted_html += text[last_idx:]
    
    return highlighted_html

# ============================================
# CHARGEMENT DU MODÈLE
# ============================================

@st.cache_resource
def load_default_model():
    """Charge le modèle par défaut"""
    try:
        default_model_path = "final_biomedical_ner_model/final_biomedical_ner_model.pth"
        default_vocabs_path = "final_biomedical_ner_model/final_biomedical_ner_model_vocabs.json"
        
        if not Path(default_model_path).exists():
            st.error(f"❌ Fichier modèle non trouvé: {default_model_path}")
            return None
        if not Path(default_vocabs_path).exists():
            st.error(f"❌ Fichier vocabs non trouvé: {default_vocabs_path}")
            return None
        
        predictor = NERPredictor(default_model_path, default_vocabs_path)
        return predictor
        
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        return None

# ============================================
# APPLICATION PRINCIPALE
# ============================================

def main():
    # Charger le modèle
    if 'predictor' not in st.session_state:
        with st.spinner("🔧 Chargement du modèle..."):
            predictor = load_default_model()
            if predictor:
                st.session_state.predictor = predictor
                st.success("✅ Modèle chargé avec succès!")
            else:
                st.error("❌ Impossible de charger le modèle")
                st.stop()
    
    if 'last_results' not in st.session_state:
        st.session_state.last_results = None
    
    predictor = st.session_state.predictor
    
    # Sidebar
    with st.sidebar:
        st.markdown("# 🧬 BioNER")
        st.markdown("**Biomedical Named Entity Recognition**")
        st.markdown("---")
        
        st.markdown("### 📊 Modèle")
        st.markdown(f"""
        - **Vocabulaire mots:** {len(predictor.word_vocab)}
        - **Vocabulaire caractères:** {len(predictor.char_vocab)}
        - **Types d'entités:** {len(predictor.label_vocab)}
        - **Dispositif:** {predictor.device}
        """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Entités Supportées")
        st.markdown("""
        - **DNA/RNA** - Séquences génétiques
        - **Protein** - Protéines et peptides
        - **Disease** - Maladies et conditions
        - **Cell** - Types et lignées cellulaires
        - **Modifier** - Modificateurs
        """)
    
    # Contenu principal
    st.title("🧬 Biomedical Named Entity Recognition")
    st.markdown("Extrayez automatiquement les entités biomédicales de vos textes")
    
    # Légende des entités - CORRIGÉ
    st.markdown("### 🎨 Types d'Entités")
    
    # Utiliser la nouvelle fonction qui utilise st.columns
    create_entity_legend()
    
    st.markdown("---")
    
    # Sélection de la méthode d'entrée
    st.markdown("### 📝 Entrée du Texte")
    
    input_method = st.radio(
        "Choisissez la méthode d'entrée:",
        ["✍️ Saisir du texte", "📁 Uploader un fichier"],
        horizontal=True
    )
    
    input_text = ""
    
    if input_method == "✍️ Saisir du texte":
        # Exemples
        examples = {
            "🧬 Génétique": "BRCA1 mutations are responsible for most cases of inherited breast and ovarian cancer. The BRCA1 protein functions as a tumour suppressor.",
            "🩸 Immunologie": "IL-2 gene expression and NF-kappa B activation through CD28 requires reactive oxygen production by 5-lipoxygenase in T cells.",
        }
        
        col1, col2 = st.columns(2)
        # Initialisation
        if 'selected_example' not in st.session_state:
            st.session_state.selected_example = ""

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🧬 Exemple Génétique", use_container_width=True):
                st.session_state.selected_example = examples["🧬 Génétique"]
        with col2:
            if st.button("🩸 Exemple Immunologie", use_container_width=True):
                st.session_state.selected_example = examples["🩸 Immunologie"]

        # Affichage du text_area
        input_text = st.text_area(
            "**Texte à analyser:**",
            value=st.session_state.selected_example,
            height=200,
            placeholder="Collez votre texte biomédical ici..."
        )

    
    else:
        uploaded_file = st.file_uploader(
            "**Choisissez un fichier texte (.txt):**",
            type=['txt']
        )
        
        if uploaded_file is not None:
            try:
                input_text = uploaded_file.read().decode('utf-8')
                st.success(f"✅ Fichier chargé")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")
    
    # Bouton d'extraction
    if st.button("🔍 Extraire les Entités", type="primary", use_container_width=True):
        if not input_text.strip():
            st.error("❌ Veuillez saisir du texte.")
        else:
            with st.spinner("🔬 Analyse en cours..."):
                start_time = time.time()
                
                try:
                    predictions = predictor.predict(input_text)
                    entities = predictor.extract_entities(predictions)
                    
                    processing_time = time.time() - start_time
                    
                    st.session_state.last_results = {
                        'predictions': predictions,
                        'entities': entities,
                        'text': input_text,
                        'processing_time': processing_time,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.success(f"✅ {len(entities)} entités trouvées en {processing_time:.2f} secondes!")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
    
    # Afficher les résultats
    if st.session_state.last_results:
        results = st.session_state.last_results
        
        # Métriques
        st.markdown("### 📊 Résultats")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Entités Trouvées", len(results['entities']))
        
        with col2:
            st.metric("Temps d'Analyse", f"{results['processing_time']:.2f}s")
        
        with col3:
            unique_types = len(set([e['type'] for e in results['entities']]))
            st.metric("Types d'Entités", unique_types)
        
        # Onglets
        tab1, tab2 = st.tabs(["📄 Texte Annoté", "📊 Détail des Entités"])
        
        with tab1:
            st.markdown("#### Texte avec Entités Surlignées")
            st.markdown(
                f"""
                <div style="background-color: #1e293b; padding: 20px; border-radius: 8px; border-left: 4px solid #3b82f6; margin: 10px 0; line-height: 1.6;">
                {highlight_text_with_entities(results['text'], results['entities'])}</div>
                """,
                unsafe_allow_html=True
            )
        
        with tab2:
            if results['entities']:
                df_data = []
                for entity in results['entities']:
                    df_data.append({
                        'Entité': entity['text'],
                        'Type': ENTITY_NAMES.get(entity['start_tag'], entity['type']),
                        'Tokens': len(entity['tokens'])
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Graphique
                type_counts = df['Type'].value_counts()
                fig = px.bar(
                    x=type_counts.index,
                    y=type_counts.values,
                    title="Distribution des Types d'Entités",
                    labels={'x': 'Type', 'y': 'Nombre'}
                )
                fig.update_layout(
                    plot_bgcolor='#1e293b',
                    paper_bgcolor='#1e293b',
                    font_color='#e2e8f0',
                    title_font_color='#60a5fa'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ Aucune entité trouvée dans le texte.")
        
        # Export
        st.markdown("### 💾 Exporter les Résultats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            json_data = {
                'text': results['text'],
                'entities': results['entities'],
                'timestamp': results['timestamp']
            }
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Télécharger JSON",
                data=json_str,
                file_name="bio_ner_results.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            if results['entities']:
                df_data = []
                for entity in results['entities']:
                    df_data.append({
                        'entite': entity['text'],
                        'type': ENTITY_NAMES.get(entity['start_tag'], entity['type'])
                    })
                
                df = pd.DataFrame(df_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Télécharger CSV",
                    data=csv,
                    file_name="bio_ner_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()