import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import numpy as np
import os

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
### loading data
def load_jnlpba_dataset(base_path):
    """
    Charge le dataset JNLPBA avec ses fichiers train/devel/test
    """
    print(f"Chargement du dataset JNLPBA depuis: {base_path}")
    
    all_sentences = []
    # Charger tous les fichiers .tsv
    for filename in ['train.tsv', 'devel.tsv', 'test.tsv']:
        file_path = os.path.join(base_path, filename)
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            sentence = []
            for line in f:
                line = line.strip()
                if not line:
                    if sentence:
                        all_sentences.append(sentence)
                        sentence = []
                    continue
                
                if line.startswith('-DOCSTART-'):
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 2:
                    token, label = parts[0], parts[1]
                    sentence.append((token, label))
            
            if sentence:
                all_sentences.append(sentence)
    
    # Chargement des classes
    classes = ['B-DNA', 'I-DNA', 'B-cell_line', 'I-cell_line', 
               'B-protein', 'I-protein', 'B-cell_type', 'I-cell_type',
               'B-RNA', 'I-RNA', 'O']
    
    print(f"- sentences: {len(all_sentences)} phrases")
    print(f"- {len(classes)} Classes: {classes}")
    
    return all_sentences, classes

def load_ncbi_dataset(folder_path):
    """
    Charge le deuxième dataset NCBI avec les annotations XML-like
    """
    print(f"Chargement du dataset NCBI depuis: {folder_path}")
    
    data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path,filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Format: ID\tTitre\tTexte
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                    
                doc_id, title, text = parts[0], parts[1], parts[2]
                
                # Extraction des entités annotées
                entities = []
                offset = 0
                
                # Recherche des annotations
                pattern = r'<category="([^"]+)">([^<]+)</category>'
                for match in re.finditer(pattern, text):
                    category = match.group(1)
                    mention = match.group(2)
                    start = match.start() - offset
                    end = match.end() - offset
                    
                    entities.append({
                        'start': start,
                        'end': end,
                        'text': mention,
                        'type': category
                    })
                    
                    # Mise à jour de l'offset pour les balises supprimées
                    offset += len(match.group(0)) - len(mention)
                
                # Nettoyage du texte (suppression des balises)
                clean_text = re.sub(pattern, r'\2', text)
                
                data.append({
                    'id': doc_id,
                    'title': title,
                    'text': clean_text,
                    'entities': entities
                })
    
    print(f"Documents chargés: {len(data)}")
    print(f"Exemple d'entités dans le premier document: {len(data[0]['entities']) if data else 0}")
    
    return data


def prepare_ncbi_for_ner(ncbi_data):
    """
    Convertit NCBI Disease Corpus en format identique à JNLPBA :
    List[Tuple[List[tokens], List[BIO-tags]]]
    """
    sentences = []

    for doc in ncbi_data:
        text = doc["text"]
        entities = doc["entities"]

        # Trier les entités par offset
        entities = sorted(entities, key=lambda e: e["start"])

        # Découpage en phrases
        try:
            raw_sentences = sent_tokenize(text)
        except:
            raw_sentences = [text]

        char_offset = 0  # offset global dans le document

        for sent in raw_sentences:
            sent = sent.strip()
            if not sent:
                char_offset += len(sent) + 1
                continue

            # Tokenisation avec positions locales
            tokens = []
            spans = []
            start = 0

            for match in re.finditer(r"\S+", sent):
                tok = match.group()
                tok_start = match.start()
                tok_end = match.end()
                tokens.append(tok)
                spans.append((tok_start, tok_end))

            labels = ["O"] * len(tokens)

            sent_start = char_offset
            sent_end = sent_start + len(sent)

            # Projection BIO exacte
            for ent in entities:
                if ent["end"] <= sent_start or ent["start"] >= sent_end:
                    continue

                for i, (tok_start, tok_end) in enumerate(spans):
                    abs_start = sent_start + tok_start
                    abs_end = sent_start + tok_end

                    if abs_start >= ent["start"] and abs_end <= ent["end"]:
                        if abs_start == ent["start"]:
                            labels[i] = f"B-{ent['type']}"
                        else:
                            labels[i] = f"I-{ent['type']}"

            # Format JNLPBA : Tuple de deux listes
            sentences.append((tokens, labels))
            char_offset += len(sent) + 1

    print(f"Total de phrases générées (format JNLPBA): {len(sentences)}")
    
    # Vérification du format
    if sentences:
        print("\nVérification du format:")
        print(f"Type du premier élément: {type(sentences[0])}")
        print(f"Longueur du tuple: {len(sentences[0])}")
        
        tokens, labels = sentences[0]
        print(f"Type tokens: {type(tokens)} (longueur: {len(tokens)})")
        print(f"Type labels: {type(labels)} (longueur: {len(labels)})")
        print(f"Exemple tokens[:5]: {tokens[:5]}")
        print(f"Exemple labels[:5]: {labels[:5]}")
    
    return sentences



def train_word2vec_embeddings(sentences, vector_size=200, window=5, min_count=2, workers=4):
    """
    Entraîne un modèle Word2Vec sur vos données
    sentences: Liste de phrases (liste de tokens)
    """
    # Extraire les tokens de toutes les phrases
    tokenized_sentences = []
    for tokens, _ in sentences:
        # Convertir en minuscules pour l'entraînement
        lower_tokens = [token.lower() for token in tokens]
        tokenized_sentences.append(lower_tokens)
    
    print(f"Nombre de phrases pour Word2Vec: {len(tokenized_sentences)}")
    print(f"Première phrase: {tokenized_sentences[0][:10]}...")
    
    # Entraîner Word2Vec
    print("Entraînement du modèle Word2Vec...")
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,  # Skip-gram (1) ou CBOW (0)
        epochs=10  # Nombre d'itérations
    )
    
    print(f"Vocabulaire Word2Vec: {len(model.wv)} mots")
    print(f"Taille des vecteurs: {vector_size}")
    
    return model

def save_word2vec_model(model, filepath):
    """
    Sauvegarde le modèle Word2Vec
    """
    # S'assurer que le répertoire existe
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Création du répertoire: {directory}")
    
    # Ajouter l'extension .model si non présente
    if not filepath.endswith('.model'):
        filepath += '.model'
    
    model.save(filepath)
    print(f"Modèle Word2Vec sauvegardé: {filepath}")
    return filepath

def load_word2vec_model(filepath):
    """
    Charge un modèle Word2Vec
    """
    # Essayer différentes extensions
    possible_paths = [
        filepath,
        filepath + '.model',
        filepath + '.bin'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = Word2Vec.load(path)
                print(f"Modèle Word2Vec chargé: {path}")
                return model
            except Exception as e:
                print(f"Erreur lors du chargement de {path}: {e}")
                continue
    
    print(f"Modèle Word2Vec non trouvé: {filepath}")
    return None

def create_embedding_matrix_from_word2vec(word2vec_model, vocab, vector_size=200):
    """
    Crée une matrice d'embeddings alignée avec votre vocabulaire
    """
    embedding_matrix = np.zeros((len(vocab), vector_size))
    
    words_found = 0
    words_not_found = 0
    
    for word, idx in vocab.items():
        if word == '<PAD>':
            # Vecteur nul pour le padding
            embedding_matrix[idx] = np.zeros(vector_size)
        elif word == '<UNK>':
            # Vecteur aléatoire pour les mots inconnus
            embedding_matrix[idx] = np.random.normal(scale=0.1, size=(vector_size,))
        elif word == '<NUM>':
            # Vecteur spécial pour les nombres
            embedding_matrix[idx] = np.random.normal(scale=0.05, size=(vector_size,))
        else:
            # Chercher le mot dans Word2Vec
            word_lower = word.lower()
            if word_lower in word2vec_model.wv:
                embedding_matrix[idx] = word2vec_model.wv[word_lower]
                words_found += 1
            else:
                # Mot non trouvé -> vecteur aléatoire
                embedding_matrix[idx] = np.random.normal(scale=0.1, size=(vector_size,))
                words_not_found += 1
    
    print(f"Mots trouvés dans Word2Vec: {words_found}")
    print(f"Mots non trouvés: {words_not_found}")
    print(f"Couverture: {words_found/(words_found+words_not_found)*100:.2f}%")
    
    return embedding_matrix