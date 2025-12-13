import os
import re

### loading data
def load_jnlpba_dataset(base_path):
    """
    Charge le dataset JNLPBA avec ses 4 fichiers
    """
    print(f"Chargement du dataset JNLPBA depuis: {base_path}")
    
    # Chargement des fichiers
    all_sentences = []
    for filename in os.listdir(base_path):
        if filename.endswith(".tsv"):
            with open(os.path.join(base_path, 'train.tsv'), 'r', encoding='utf-8') as f:
                sentence = []
                for line in f:
                    line = line.strip()
                    if not line:
                        if sentence:
                            all_sentences.append(sentence)
                            sentence = []
                        continue
                    
                    # Ignorer -DOCSTART-
                    if line.startswith('-DOCSTART-'):
                        continue
                        
                    token, label = line.split('\t')
                    sentence.append((token, label))
                
                if sentence:
                    all_sentences.append(sentence)
    
    # Chargement des classes
    with open(os.path.join(base_path, 'classes.txt'), 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    print(f"- sentences: {len(all_sentences)} phrases")
    print(f"- Classes: {classes}")
    
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
    Prépare les données NCBI au format BIO pour le NER
    """
    sentences = []
    
    for doc in ncbi_data:
        text = doc['text']
        entities = doc['entities']
        
        # Segmentation en phrases approximative
        sentence_boundaries = []
        start = 0
        
        for match in re.finditer(r'[.!?]\s+', text):
            end = match.end()
            sentence_boundaries.append((start, end))
            start = end
        
        if start < len(text):
            sentence_boundaries.append((start, len(text)))
        
        # Pour chaque phrase, créer des annotations BIO
        for sent_start, sent_end in sentence_boundaries:
            sentence_text = text[sent_start:sent_end]
            tokens = sentence_text.split()
            
            # Initialiser tous les tokens comme O
            labels = ['O'] * len(tokens)
            
            # Marquer les entités dans cette phrase
            for entity in entities:
                entity_start = entity['start']
                entity_end = entity['end']
                entity_text = entity['text']
                entity_type = entity['type']
                
                # Vérifier si l'entité est dans cette phrase
                if entity_start >= sent_start and entity_end <= sent_end:
                    # Trouver les tokens couverts par l'entité
                    entity_tokens = entity_text.split()
                    
                    # Recherche approximative des tokens
                    for i in range(len(tokens) - len(entity_tokens) + 1):
                        if tokens[i:i+len(entity_tokens)] == entity_tokens:
                            # Marquer comme B-XXX pour le premier token, I-XXX pour les suivants
                            labels[i] = f'B-{entity_type}'
                            for j in range(1, len(entity_tokens)):
                                labels[i+j] = f'I-{entity_type}'
                            break
            
            # Ajouter la phrase avec ses labels
            sentence_with_labels = [(token, label) for token, label in zip(tokens, labels)]
            sentences.append(sentence_with_labels)
    
    return sentences