from collections import Counter

def create_vocab(sentences, min_freq=2):
    """
    Crée un vocabulaire à partir des phrases
    Supporte deux formats:
    1. Liste de tuples (tokens, labels) - format JNLPBA
    2. Liste de listes de tuples (token, label) - format NCBI original
    """
    from collections import Counter
    
    word_counts = Counter()
    
    # Vérifier le format des données
    if not sentences:
        return {'<PAD>': 0, '<UNK>': 1, '<NUM>': 2}
    
    first_item = sentences[0]
    
    if isinstance(first_item, tuple) and len(first_item) == 2:
        # Format 1: (tokens, labels)
        print("Format vocab: Tuple (tokens, labels)")
        for tokens, _ in sentences:
            if isinstance(tokens, list):
                word_counts.update([token.lower() for token in tokens])
            else:
                print(f"Attention: tokens n'est pas une liste: {type(tokens)}")
                
    elif isinstance(first_item, list):
        # Format 2: Liste de (token, label)
        print("Format vocab: Liste de paires (token, label)")
        for sentence in sentences:
            if isinstance(sentence, list):
                for item in sentence:
                    if isinstance(item, tuple) and len(item) == 2:
                        token, _ = item
                        word_counts.update([token.lower()])
            else:
                print(f"Attention: élément non liste: {type(sentence)}")
    else:
        print(f"Format non reconnu: {type(first_item)}")
        raise ValueError(f"Format de données non supporté: {type(first_item)}")
    
    # Créer le vocabulaire
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<NUM>': 2
    }
    
    # Ajouter les mots avec fréquence minimale
    idx = 3
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    print(f"Vocabulaire créé: {len(vocab)} mots")
    print(f"Mots uniques: {len(word_counts)}")
    print(f"Mots avec fréquence >= {min_freq}: {idx - 3}")
    
    return vocab


def create_char_vocab(sentences):
    """Crée le vocabulaire de caractères - Supporte deux formats"""
    char_counts = Counter()
    
    # Vérifier le format des données
    if not sentences:
        return {'<PAD>': 0, '<UNK>': 1}
    
    first_item = sentences[0]
    
    if isinstance(first_item, tuple) and len(first_item) == 2:
        # Format 1: (tokens, labels)
        print("Format char vocab: Tuple (tokens, labels)")
        for tokens, _ in sentences:
            if isinstance(tokens, list):
                for token in tokens:
                    char_counts.update(token)
            else:
                print(f"Attention: tokens n'est pas une liste: {type(tokens)}")
                
    elif isinstance(first_item, list):
        # Format 2: Liste de (token, label)
        print("Format char vocab: Liste de paires (token, label)")
        for sentence in sentences:
            if isinstance(sentence, list):
                for item in sentence:
                    if isinstance(item, tuple) and len(item) == 2:
                        token, _ = item
                        char_counts.update(token)
            else:
                print(f"Attention: élément non liste: {type(sentence)}")
    else:
        print(f"Format non reconnu: {type(first_item)}")
        raise ValueError(f"Format de données non supporté: {type(first_item)}")
    
    # Créer le vocabulaire de caractères
    char_vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, char in enumerate(char_counts.keys(), start=2):
        char_vocab[char] = idx
    
    print(f"Vocabulaire caractères créé: {len(char_vocab)} caractères")
    print(f"Caractères uniques: {len(char_counts)}")
    
    return char_vocab

def preprocess_tokens(tokens, vocab):
    """Prétraite les tokens : convertit en IDs, gère UNK et NUM"""
    processed = []
    for token in tokens:
        # Remplacer les nombres par <NUM>
        if token.isdigit():
            token = '<NUM>'
        # Convertir en minuscule et obtenir l'ID
        token_lower = token.lower()
        token_id = vocab.get(token_lower, vocab['<UNK>'])
        processed.append(token_id)
    return processed

def create_char_sequences(tokens, char_vocab, max_word_len=20):
    """Convertit les tokens en séquences de caractères"""
    char_seqs = []
    for token in tokens:
        char_seq = [char_vocab.get(c, char_vocab['<UNK>']) for c in token[:max_word_len]]
        # Pad si nécessaire
        if len(char_seq) < max_word_len:
            char_seq += [char_vocab['<PAD>']] * (max_word_len - len(char_seq))
        char_seqs.append(char_seq)
    return char_seqs

def create_tag_mapping(sentences):
    """Crée le mapping des tags - Supporte deux formats"""
    all_tags = set()
    
    # Vérifier le format des données
    if not sentences:
        return {'<PAD>': 0}, {0: '<PAD>'}
    
    first_item = sentences[0]
    
    if isinstance(first_item, tuple) and len(first_item) == 2:
        # Format 1: (tokens, labels)
        print("Format tag mapping: Tuple (tokens, labels)")
        for _, labels in sentences:
            if isinstance(labels, list):
                all_tags.update(labels)
            else:
                print(f"Attention: labels n'est pas une liste: {type(labels)}")
                
    elif isinstance(first_item, list):
        # Format 2: Liste de (token, label)
        print("Format tag mapping: Liste de paires (token, label)")
        for sentence in sentences:
            if isinstance(sentence, list):
                for item in sentence:
                    if isinstance(item, tuple) and len(item) == 2:
                        _, label = item
                        all_tags.add(label)
            else:
                print(f"Attention: élément non liste: {type(sentence)}")
    else:
        print(f"Format non reconnu: {type(first_item)}")
        raise ValueError(f"Format de données non supporté: {type(first_item)}")
    
    # Créer le mapping
    tag_to_idx = {'<PAD>': 0}
    for tag in sorted(all_tags):
        if tag not in tag_to_idx:
            tag_to_idx[tag] = len(tag_to_idx)
    
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
    
    print(f"Mapping tags créé: {len(tag_to_idx)} tags uniques")
    print(f"Tags: {sorted(all_tags)}")
    
    return tag_to_idx, idx_to_tag