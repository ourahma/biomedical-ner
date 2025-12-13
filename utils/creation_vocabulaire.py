from collections import Counter, defaultdict

def build_vocabularies(data, max_vocab_size=20000):
    """
    Construit les vocabulaires pour les mots et caractères
    """
    print("Construction des vocabulaires...")
    
    # Comptage des mots
    word_counter = Counter()
    char_counter = Counter()
    
    for sentence in data:
        for token, _ in sentence:
            word_counter[token.lower()] += 1
            for char in token:
                char_counter[char] += 1
    
    # Création du vocabulaire mots
    word_vocab = {'<PAD>': 0, '<UNK>': 1, '<NUM>': 2}
    
    # Ajout des mots les plus fréquents
    for i, (word, count) in enumerate(word_counter.most_common(max_vocab_size - 3)):
        word_vocab[word] = i + 3
    
    # Création du vocabulaire caractères
    char_vocab = {'<PAD>': 0, '<UNK>': 1}
    
    for i, (char, count) in enumerate(char_counter.most_common(200)):  # Limiter à 200 caractères
        char_vocab[char] = i + 2
    
    print(f"  - Taille vocabulaire mots: {len(word_vocab)}")
    print(f"  - Taille vocabulaire caractères: {len(char_vocab)}")
    
    return word_vocab, char_vocab

def build_label_vocab(data):
    """
    Construit le vocabulaire des labels BIO
    """
    label_counter = Counter()
    
    for sentence in data:
        for _, label in sentence:
            label_counter[label] += 1
    
    label_vocab = {}
    for label in label_counter:
        label_vocab[label] = len(label_vocab)
    
    # Ajouter l'index pour le padding si nécessaire
    if '<PAD>' not in label_vocab:
        label_vocab['<PAD>'] = len(label_vocab)
    
    print(f"  - Taille vocabulaire labels: {len(label_vocab)}")
    print(f"  - Labels: {list(label_vocab.keys())}")
    
    return label_vocab