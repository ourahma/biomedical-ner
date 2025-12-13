from torch.utils.data import Dataset, DataLoader
import re
import torch


class NERDataset(Dataset):
    """
    Dataset personnalisé pour le NER biomédical
    """
    def __init__(self, data, word_vocab, char_vocab, label_vocab, max_word_len=20, max_sent_len=100):
        self.data = data
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.label_vocab = label_vocab
        self.max_word_len = max_word_len
        self.max_sent_len = max_sent_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sentence = self.data[idx]
        
        # Limiter la longueur de la phrase
        if len(sentence) > self.max_sent_len:
            sentence = sentence[:self.max_sent_len]
        
        # Préparer les séquences
        word_ids = []
        char_ids = []
        label_ids = []
        
        for token, label in sentence:
            # Encodage du mot
            word_lower = token.lower()
            
            # Remplacement des nombres
            if re.match(r'^\d+$', token):
                word_id = self.word_vocab.get('<NUM>', self.word_vocab['<UNK>'])
            else:
                word_id = self.word_vocab.get(word_lower, self.word_vocab['<UNK>'])
            
            word_ids.append(word_id)
            
            # Encodage des caractères
            token_chars = []
            for char in token[:self.max_word_len]:
                token_chars.append(self.char_vocab.get(char, self.char_vocab['<UNK>']))
            
            # Padding pour les caractères
            while len(token_chars) < self.max_word_len:
                token_chars.append(self.char_vocab['<PAD>'])
            
            char_ids.append(token_chars)
            
            # Encodage du label
            label_ids.append(self.label_vocab[label])
        
        # Padding de la phrase
        while len(word_ids) < self.max_sent_len:
            word_ids.append(self.word_vocab['<PAD>'])
            char_ids.append([self.char_vocab['<PAD>']] * self.max_word_len)
            label_ids.append(self.label_vocab['<PAD>'])
        
        return {
            'words': torch.tensor(word_ids, dtype=torch.long),
            'chars': torch.tensor(char_ids, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'length': min(len(sentence), self.max_sent_len)
        }