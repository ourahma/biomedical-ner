import os
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize
from gensim.models import FastText
from collections import Counter

# =========================
# NLTK SETUP
# =========================
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# =========================
# LOAD JNLPBA DATASET
# =========================
def load_jnlpba_dataset(base_path):
    print(f"Chargement du dataset JNLPBA depuis: {base_path}")

    all_sentences = []

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

    classes = [
        'B-DNA', 'I-DNA',
        'B-cell_line', 'I-cell_line',
        'B-protein', 'I-protein',
        'B-cell_type', 'I-cell_type',
        'B-RNA', 'I-RNA',
        'O'
    ]

    print(f"- sentences: {len(all_sentences)}")
    print(f"- classes: {classes}")

    return all_sentences, classes


# =========================
# LOAD NCBI DATASET
# =========================
def load_ncbi_dataset(folder_path):
    print(f"Chargement du dataset NCBI depuis: {folder_path}")

    data = []
    pattern = r'<category="([^"]+)">([^<]+)</category>'

    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) < 3:
                    continue

                doc_id, title, text = parts[0], parts[1], parts[2]

                entities = []
                offset = 0

                for match in re.finditer(pattern, text):
                    category = match.group(1)
                    mention = match.group(2)

                    start = match.start() - offset
                    end = match.end() - offset

                    entities.append({
                        "start": start,
                        "end": end,
                        "text": mention,
                        "type": category
                    })

                    offset += len(match.group(0)) - len(mention)

                clean_text = re.sub(pattern, r'\2', text)

                data.append({
                    "id": doc_id,
                    "title": title,
                    "text": clean_text,
                    "entities": entities
                })

    print(f"Documents chargés: {len(data)}")
    return data


# =========================
# NCBI → JNLPBA FORMAT
# =========================
def prepare_ncbi_for_ner(ncbi_data):
    sentences = []

    for doc in ncbi_data:
        text = doc["text"]
        entities = sorted(doc["entities"], key=lambda x: x["start"])

        try:
            raw_sentences = sent_tokenize(text)
        except:
            raw_sentences = [text]

        char_offset = 0

        for sent in raw_sentences:
            sent = sent.strip()
            if not sent:
                char_offset += len(sent) + 1
                continue

            tokens = []
            spans = []

            for match in re.finditer(r"\S+", sent):
                tokens.append(match.group())
                spans.append((match.start(), match.end()))

            labels = ["O"] * len(tokens)

            sent_start = char_offset
            sent_end = sent_start + len(sent)

            for ent in entities:
                if ent["end"] <= sent_start or ent["start"] >= sent_end:
                    continue

                for i, (tok_start, tok_end) in enumerate(spans):
                    abs_start = sent_start + tok_start
                    abs_end = sent_start + tok_end

                    # ✅❌ Check if token overlaps with entity

                    if abs_start >= ent["start"] and abs_end <= ent["end"]:
                        if abs_start == ent["start"]:
                            labels[i] = f"B-{ent['type']}"
                        else:
                            labels[i] = f"I-{ent['type']}"

            sentences.append((tokens, labels))
            char_offset += len(sent) + 1

    return sentences


# =========================
# TRAIN FASTTEXT EMBEDDINGS
# =========================
def train_fasttext_embeddings(
    sentences,
    vector_size=200,
    window=5,
    min_count=2,
    workers=4,
    min_n=3,
    max_n=6
):
    tokenized_sentences = [
        [token.lower() for token, _ in sent]
        for sent in sentences
    ]

    print(f"Phrases pour FastText: {len(tokenized_sentences)}")
    print(f"Exemple: {tokenized_sentences[0][:10]}")

    model = FastText(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,              # Skip-gram
        min_n=min_n,
        max_n=max_n,
        epochs=10
    )

    print(f"Vocabulaire FastText: {len(model.wv)}")
    return model


# =========================
# SAVE / LOAD FASTTEXT
# =========================
def save_fasttext_model(model, filepath):
    if not filepath.endswith(".model"):
        filepath += ".model"

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"FastText sauvegardé: {filepath}")
    return filepath


def load_fasttext_model(filepath):
    if not filepath.endswith(".model"):
        filepath += ".model"

    if not os.path.exists(filepath):
        print("Modèle introuvable.")
        return None

    model = FastText.load(filepath)
    print(f"FastText chargé: {filepath}")
    return model


# =========================
# CREATE EMBEDDING MATRIX
# =========================
# ✅❌ Add dataset validation
def create_embedding_matrix_from_fasttext(model, vocab, vector_size=200):
    embedding_matrix = np.zeros((len(vocab), vector_size))
    
    oov_words = []
    oov_count = 0
    
    for word, idx in vocab.items():
        if word == "<PAD>":
            embedding_matrix[idx] = np.zeros(vector_size)
        elif word in ["<UNK>", "<NUM>"]:
            # Better initialization for special tokens
            embedding_matrix[idx] = np.random.uniform(-0.1, 0.1, size=vector_size)
        else:
            # FastText handles OOV via subwords, but we should track it
            embedding_matrix[idx] = model.wv[word.lower()]
            if word.lower() not in model.wv:
                oov_words.append(word)
                oov_count += 1
    
    if oov_count > 0:
        print(f"Warning: {oov_count} words not in FastText vocabulary")
        if oov_count < 20:
            print(f"OOV words: {oov_words}")
    
    return embedding_matrix


# =========================
# OPTIONAL: DATASET ANALYSIS
# =========================
def visualize_dataset_distribution(results, dataset_name="JNLPBA"):
    print("=" * 60)
    print(f"ANALYSE DATASET: {dataset_name}")
    print("=" * 60)

    splits = [s for s in ['train', 'dev', 'test'] if f"{s}_sentences" in results]

    for split in splits:
        sentences = results[f"{split}_sentences"]
        tokens = sum(len(s) for s in sentences)
        entities = sum(1 for s in sentences for _, t in s if t != "O")

        print(f"\n{split.upper()}")
        print(f"  phrases: {len(sentences)}")
        print(f"  tokens: {tokens}")
        print(f"  entités: {entities}")

    print("=" * 60)



# ✅❌ Add comprehensive dataset statistics:
def analyze_dataset_statistics(sentences, dataset_name="JNLPBA"):
    """Detailed analysis of NER dataset"""
    total_tokens = 0
    entity_counts = Counter()
    sentence_lengths = []
    
    for tokens, labels in sentences:
        total_tokens += len(tokens)
        sentence_lengths.append(len(tokens))
        
        current_entity = None
        for label in labels:
            if label != 'O':
                entity_counts[label] += 1
    
    print(f"\n=== {dataset_name} Statistics ===")
    print(f"Total sentences: {len(sentences)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Avg sentence length: {np.mean(sentence_lengths):.1f}")
    print(f"Entities distribution:")
    for entity, count in entity_counts.most_common():
        print(f"  {entity}: {count}")