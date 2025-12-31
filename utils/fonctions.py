import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


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
    print(f"Chargement du dataset NCBI depuis: {folder_path}")

    data = []

    tag_pattern = re.compile(r'<category="([^"]+)">([^<]+)</category>')

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 3:
                    continue

                doc_id, title, text = parts[0], parts[1], parts[2]

                entities = []
                clean_text = ""
                last_end = 0

                # Reconstruire le texte sans balises en recalculant les offsets
                for match in tag_pattern.finditer(text):
                    start, end = match.span()
                    mention = match.group(2)

                    clean_text += text[last_end:start]
                    ent_start = len(clean_text)
                    clean_text += mention
                    ent_end = len(clean_text)

                    entities.append({
                        "start": ent_start,
                        "end": ent_end,
                        "type": "Disease"   
                    })

                    last_end = end

                clean_text += text[last_end:]

                data.append({
                    "id": doc_id,
                    "title": title,
                    "text": clean_text,
                    "entities": entities
                })

    print(f"Documents chargés: {len(data)}")
    print(f"Entités dans le premier document: {len(data[0]['entities']) if data else 0}")

    return data


def prepare_ncbi_for_ner(ncbi_data):
    """
    Convertit NCBI Disease Corpus vers :
    List[Tuple[List[str], List[str]]] en BIO (B-Disease / I-Disease / O)
    """
    sentences = []

    token_pattern = re.compile(r"\w+|[^\w\s]")

    for doc in ncbi_data:
        text = doc["text"]
        entities = doc["entities"]

        tokens = []
        spans = []

        # Tokenisation avec offsets stables
        for match in token_pattern.finditer(text):
            tokens.append(match.group())
            spans.append((match.start(), match.end()))

        labels = ["O"] * len(tokens)

        # Projection BIO par overlap
        for ent in entities:
            ent_start, ent_end = ent["start"], ent["end"]
            first_token = True

            for i, (tok_start, tok_end) in enumerate(spans):
                overlaps = not (tok_end <= ent_start or tok_start >= ent_end)

                if overlaps:
                    if first_token:
                        labels[i] = "B-Disease"
                        first_token = False
                    else:
                        labels[i] = "I-Disease"

        sentences.append((tokens, labels))

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

def visualize_dataset_distribution(results, dataset_name="Dataset"):
    """
    Affiche des statistiques détaillées sur le dataset
    
    Args:
        results: Dictionnaire retourné par create_jnlpba_dataloaders
        dataset_name: Nom du dataset pour le titre des graphiques
    """
    
    # Configuration des couleurs
    colors = {'train': 'skyblue', 'dev': 'orange', 'test': 'green'}
    
    # 1. Informations de base
    print("=" * 60)
    print(f"ANALYSE DU DATASET: {dataset_name}")
    print("=" * 60)
    
    # Vérifier que les clés existent
    splits = []
    for split in ['train', 'dev', 'test']:
        if f'{split}_sentences' in results:
            splits.append(split)
    
    if not splits:
        print("ERREUR: Aucune donnée trouvée dans 'results'")
        return
    
    # 2. Statistiques par split
    print("\n1. RÉPARTITION DES DONNÉES")
    print("-" * 40)
    
    stats = {}
    for split in splits:
        sentences = results[f'{split}_sentences']
        num_sentences = len(sentences)
        num_tokens = sum(len(sent) for sent in sentences)
        num_entities = sum(1 for sent in sentences for _, tag in sent if tag != 'O')
        
        stats[split] = {
            'sentences': num_sentences,
            'tokens': num_tokens,
            'entities': num_entities
        }
        
        print(f"\n{split.upper()}:")
        print(f"  Phrases: {num_sentences:,}")
        print(f"  Tokens: {num_tokens:,}")
        print(f"  Entités nommées: {num_entities:,}")
        if num_tokens > 0:
            print(f"  Densité d'entités: {num_entities/num_tokens*100:.1f}%")
    
    # 3. Longueur des phrases
    print("\n2. LONGUEUR DES PHRASES")
    print("-" * 40)
    
    fig, axes = plt.subplots(1, len(splits), figsize=(15, 4))
    if len(splits) == 1:
        axes = [axes]
    
    for idx, split in enumerate(splits):
        sentences = results[f'{split}_sentences']
        lengths = [len(sent) for sent in sentences]
        
        # Calculer des statistiques
        mean_len = np.mean(lengths)
        median_len = np.median(lengths)
        max_len = np.max(lengths)
        min_len = np.min(lengths)
        
        print(f"\n{split.upper()}:")
        print(f"  Moyenne: {mean_len:.1f} tokens")
        print(f"  Médiane: {median_len:.1f} tokens")
        print(f"  Min-Max: {min_len}-{max_len} tokens")
        print(f"  >100 tokens: {sum(1 for l in lengths if l > 100):,}")
        
        # Histogramme
        ax = axes[idx]
        ax.hist(lengths, bins=30, color=colors[split], edgecolor='black', alpha=0.7)
        ax.axvline(mean_len, color='red', linestyle='--', label=f'Moyenne: {mean_len:.1f}')
        ax.axvline(median_len, color='green', linestyle='--', label=f'Médiane: {median_len:.1f}')
        ax.set_xlabel('Nombre de tokens')
        ax.set_ylabel('Nombre de phrases')
        ax.set_title(f'Longueur des phrases - {split}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Distribution des longueurs - {dataset_name}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 4. Distribution des classes
    print("\n3. DISTRIBUTION DES CLASSES D'ENTITÉS")
    print("-" * 40)
    
    # Collecter toutes les classes
    all_classes = set()
    class_distributions = {}
    
    for split in splits:
        sentences = results[f'{split}_sentences']
        labels = [tag for sent in sentences for _, tag in sent]
        counter = Counter(labels)
        class_distributions[split] = counter
        all_classes.update(counter.keys())
    
    # Trier les classes (sauf 'O' qu'on met à part)
    sorted_classes = sorted([c for c in all_classes if c != 'O'])
    if 'O' in all_classes:
        sorted_classes = ['O'] + sorted_classes
    
    # Afficher le tableau des fréquences
    print("\nFréquences absolues:")
    header = f"{'Classe':<20} " + " ".join([f"{s.upper():>10}" for s in splits])
    print(header)
    print("-" * (20 + 11 * len(splits)))
    
    for cls in sorted_classes:
        row = f"{cls:<20}"
        for split in splits:
            count = class_distributions[split].get(cls, 0)
            row += f" {count:>10,}"
        print(row)
    
    print("\nPourcentages (par split):")
    header = f"{'Classe':<20} " + " ".join([f"{s.upper():>10}" for s in splits])
    print(header)
    print("-" * (20 + 11 * len(splits)))
    
    for cls in sorted_classes:
        row = f"{cls:<20}"
        for split in splits:
            total = sum(class_distributions[split].values())
            count = class_distributions[split].get(cls, 0)
            percentage = (count / total * 100) if total > 0 else 0
            row += f" {percentage:>9.1f}%"
        print(row)
    
    # 5. Graphique des classes (sans 'O' pour plus de lisibilité)
    print("\n4. VISUALISATION DES ENTITÉS (sans 'O')")
    
    entity_classes = [c for c in sorted_classes if c != 'O']
    if entity_classes:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot groupé
        x = np.arange(len(entity_classes))
        width = 0.25
        
        for i, split in enumerate(splits):
            counts = [class_distributions[split].get(cls, 0) for cls in entity_classes]
            axes[0].bar(x + i*width, counts, width=width, 
                       color=colors[split], label=split, edgecolor='black')
        
        axes[0].set_xlabel('Classes d\'entités')
        axes[0].set_ylabel('Nombre d\'occurrences')
        axes[0].set_title(f'Distribution des entités par split - {dataset_name}')
        axes[0].set_xticks(x + width*(len(splits)-1)/2)
        axes[0].set_xticklabels(entity_classes, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Pie chart pour l'ensemble train
        if 'train' in splits:
            train_counts = [class_distributions['train'].get(cls, 0) for cls in entity_classes]
            # Filtrer les classes avec 0 occurrence
            nonzero_data = [(cls, count) for cls, count in zip(entity_classes, train_counts) if count > 0]
            if nonzero_data:
                entity_classes_filtered, train_counts_filtered = zip(*nonzero_data)
                axes[1].pie(train_counts_filtered, labels=entity_classes_filtered,
                           autopct='%1.1f%%', startangle=90)
                axes[1].set_title(f'Distribution des entités - Train split')
            else:
                axes[1].text(0.5, 0.5, 'Aucune entité trouvée\n dans le split train',
                           ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        plt.show()
    
    # 6. Analyse des entités par phrase
    print("\n5. ENTITÉS PAR PHRASE")
    print("-" * 40)
    
    fig, axes = plt.subplots(1, len(splits), figsize=(15, 4))
    if len(splits) == 1:
        axes = [axes]
    
    for idx, split in enumerate(splits):
        sentences = results[f'{split}_sentences']
        entities_per_sentence = []
        
        for sent in sentences:
            # Compter les entités (tout ce qui n'est pas 'O')
            entities = sum(1 for _, tag in sent if tag != 'O')
            entities_per_sentence.append(entities)
        
        # Statistiques
        mean_ent = np.mean(entities_per_sentence)
        median_ent = np.median(entities_per_sentence)
        sentences_without_entities = sum(1 for e in entities_per_sentence if e == 0)
        
        print(f"\n{split.upper()}:")
        print(f"  Entités/phrase (moyenne): {mean_ent:.2f}")
        print(f"  Entités/phrase (médiane): {median_ent:.1f}")
        print(f"  Phrases sans entité: {sentences_without_entities:,} ({sentences_without_entities/len(sentences)*100:.1f}%)")
        
        # Histogramme
        ax = axes[idx]
        ax.hist(entities_per_sentence, bins=20, color=colors[split], 
                edgecolor='black', alpha=0.7)
        ax.set_xlabel('Nombre d\'entités')
        ax.set_ylabel('Nombre de phrases')
        ax.set_title(f'Entités par phrase - {split}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Distribution des entités par phrase - {dataset_name}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 7. Résumé des tags BIO
    print("\n6. ANALYSE DES TAGS BIO")
    print("-" * 40)
    
    bio_stats = {'B': 0, 'I': 0, 'O': 0, 'autres': 0}
    
    for split in splits:
        sentences = results[f'{split}_sentences']
        for sent in sentences:
            for _, tag in sent:
                if tag == 'O':
                    bio_stats['O'] += 1
                elif tag.startswith('B-'):
                    bio_stats['B'] += 1
                elif tag.startswith('I-'):
                    bio_stats['I'] += 1
                else:
                    bio_stats['autres'] += 1
    
    total_tags = sum(bio_stats.values())
    print(f"Total des tags: {total_tags:,}")
    for bio_type, count in bio_stats.items():
        percentage = (count / total_tags * 100) if total_tags > 0 else 0
        print(f"  {bio_type}: {count:,} ({percentage:.1f}%)")
    
    # 8. Informations supplémentaires
    print("\n7. INFORMATIONS SUPPLÉMENTAIRES")
    print("-" * 40)
    
    # Vocabulaire
    if 'vocab' in results:
        vocab_size = len(results['vocab'])
        print(f"Taille du vocabulaire: {vocab_size:,} mots")
    
    if 'char_vocab' in results:
        char_vocab_size = len(results['char_vocab'])
        print(f"Taille du vocabulaire caractères: {char_vocab_size:,}")
    
    if 'tag_to_idx' in results:
        tag_count = len(results['tag_to_idx'])
        print(f"Nombre de classes uniques: {tag_count}")
        print(f"Classes: {list(results['tag_to_idx'].keys())}")
    
    print("\n" + "=" * 60)
    print("ANALYSE TERMINÉE")
    print("=" * 60)
    
def convert_to_standard_format(results):
    """
    Convertit les résultats du format NCBI (tuple de 2 listes) vers le format standard.
    
    Args:
        results: Dictionnaire avec des phrases au format tuple (tokens, tags)
    
    Returns:
        Nouveau dictionnaire avec les phrases au format standard [(token, tag), ...]
    """
    new_results = {}
    
    for key, value in results.items():
        if key.endswith('_sentences'):
            # Convertir les phrases
            converted_sentences = []
            print(f"\nDEBUG Conversion - Clé: {key}")
            print(f"Nombre de phrases: {len(value)}")
            
            # Afficher la première phrase avant conversion
            if value:
                first_sentence = value[0]
                print(f"\nPremière phrase avant conversion:")
                print(f"Type: {type(first_sentence)}")
                print(f"Est-ce un tuple? {isinstance(first_sentence, tuple)}")
                print(f"Longueur du tuple: {len(first_sentence)}")
                
                if isinstance(first_sentence, tuple) and len(first_sentence) == 2:
                    print(f"Premier élément (tokens): {first_sentence[0][:5]}...")  # Afficher seulement les 5 premiers
                    print(f"Second élément (tags): {first_sentence[1][:5]}...")      # Afficher seulement les 5 premiers
            
            for i, sent in enumerate(value):
                try:
                    # Format tuple (tokens, tags)
                    if isinstance(sent, tuple) and len(sent) == 2:
                        tokens, tags = sent[0], sent[1]
                        # Vérifier que les longueurs correspondent
                        if len(tokens) != len(tags):
                            print(f"WARNING: Longueurs différentes dans la phrase {i}: tokens={len(tokens)}, tags={len(tags)}")
                        
                        # Convertir en liste de tuples
                        converted_sent = list(zip(tokens, tags))
                        converted_sentences.append(converted_sent)
                        
                        # Debug pour la première phrase
                        if i == 0:
                            print(f"\nPremière phrase après conversion:")
                            print(f"Type: {type(converted_sent)}")
                            print(f"Longueur: {len(converted_sent)}")
                            print(f"Premier élément: {converted_sent[0]}")
                            print(f"Type du premier élément: {type(converted_sent[0])}")
                    
                    # Format liste [tokens, tags] (ancien format)
                    elif isinstance(sent, list) and len(sent) == 2:
                        tokens, tags = sent[0], sent[1]
                        converted_sent = list(zip(tokens, tags))
                        converted_sentences.append(converted_sent)
                    
                    else:
                        print(f"WARNING: Format inattendu pour la phrase {i}: {type(sent)}")
                        converted_sentences.append(sent)
                        
                except Exception as e:
                    print(f"ERREUR lors de la conversion de la phrase {i}: {e}")
                    print(f"Contenu: {sent}")
                    converted_sentences.append(sent)
            
            new_results[key] = converted_sentences
        else:
            # Copier les autres clés
            new_results[key] = value
    
    return new_results