
# BioNER - Biomedical Named Entity Recognition

An interactive web application for biomedical named entity recognition using deep learning models.

## 🚀 Features

- **Biomedical entity recognition** (JNLPBA dataset)
  - DNA
  - RNA
  - Proteins
  - Cell Types
  - Cell Lines

- **Disease entity recognition** (NCBI dataset)
  - Diseases

- **Complete user interface**
  - Interactive color-coded visualization
  - Detailed entity extraction tables
  - Statistics and charts
  - Export results (JSON/CSV)
  - Advanced debugging mode

- **Advanced deep learning models**
  - BiLSTM with attention mechanism
  - Character-level CNN
  - Pre-trained Word2Vec embeddings
  - Conditional Random Fields (CRF) for decoding

## 📋 Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Streamlit 1.24+
- See `requirements.txt` for complete list

## 🛠 Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/bioner.git
cd bioner
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download pre-trained models**

For JNLPBA:
- Place model in `./checkpoints/JNLPBA/WE/best_model.pt`
- Place vocabulary in `./vocab/jnlpba/`
- Place embeddings in `./word2Vecembeddings/jnlpba_word2vec`

For NCBI:
- Place model in `./checkpoints/NCBI/WE_char_bilstm_cnn_attention/best_model.pt`
- Place vocabulary in `./vocab/ncbi/`
- Place embeddings in `./word2Vecembeddings/ncbi.model`

4. **Run the application**
```bash
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
bioner/
├── streamlit_app.py              # Main application
├── models/
│   └── models.py                # Deep learning models
├── streamlit_utils.py           # Streamlit utilities
├── checkpoints/                 # Pre-trained models
│   ├── JNLPBA/
│   │   └── WE/
│   │       └── best_model.pt
│   └── NCBI/
│       └── WE_char_bilstm_cnn_attention/
│           └── best_model.pt
├── vocab/                       # Vocabularies
│   ├── jnlpba/
│   └── ncbi/
├── word2Vecembeddings/          # Pre-trained embeddings
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🎯 Usage

### 1. Biomedical NER Page (JNLPBA)

**To analyze general biomedical text:**
```python
# Example text:
"Mutations in the TP53 gene are frequently observed in human cancers."
```

**Detected entities:**
- `TP53` → Protein (B-protein)
- `gene` → Protein (I-protein)

### 2. Disease NER Page (NCBI)

**To analyze diseases:**
```python
# Example text:
"Breast cancer and ovarian cancer are common diseases."
```

**Detected entities:**
- `Breast cancer` → Disease (B-Disease, I-Disease)
- `ovarian cancer` → Disease (B-Disease, I-Disease)

### Graphical Interface

1. **Select a page** in the sidebar
2. **Paste your text** in the text area
3. **Click "Analyze text"**
4. **View results** in tabs:
   - 📄 Annotated text (highlighted entities)
   - 🔍 Raw predictions (all tokens and tags)
   - 📊 Grouped entities (extracted entities)
   - 📈 Statistics (charts and metrics)
   - 🐛 Debug details (technical information)

5. **Export results** as JSON or CSV

## 🧠 Model Architecture

### CombinatorialNER Model

```python
class CombinatorialNER(nn.Module):
    """
    Architecture combining:
    - Word Embeddings (pre-trained Word2Vec)
    - Character-level CNN/LSTM
    - BiLSTM with attention
    - Conditional Random Fields (CRF)
    """
```

**This implementation is a replication of the model from:**
> **"Combinatorial embedding based multi-channel adaptive attentive neural network for biomedical named entity recognition"**  
> *Chaoxin Yuan, Jiacheng Wang, Weidong Yang, Yin Zhang*  
> Journal of Biomedical Informatics, Volume 104, 2020  
> [https://www.sciencedirect.com/science/article/pii/S1532046420300083](https://www.sciencedirect.com/science/article/pii/S1532046420300083)

**Key features from the paper:**
- Multi-channel embedding layer (word + character embeddings)
- Adaptive attention mechanism
- Combinatorial neural network architecture
- Cross-domain biomedical NER capabilities

**JNLPBA Configuration:**
- Word Embeddings: 200 dimensions
- BiLSTM hidden dim: 256
- 12 tag classes
- Without character CNN/LSTM

**NCBI Configuration:**
- Word Embeddings: 200 dimensions
- BiLSTM hidden dim: 128
- 4 tag classes
- With character CNN and LSTM
- With attention mechanism

## 📊 Datasets & Performance

### JNLPBA (Biomedical NER)
- **Entities**: 5 types (DNA, RNA, Protein, Cell Type, Cell Line)
- **Tags**: 11 BIO tags + O + PAD = 13 classes
- **Vocabulary**: 12,664 words
- **Characters**: 85 characters

**Performance Metrics:**
```
              precision    recall  f1-score   support

       B-DNA       0.72      0.68      0.70       857
       B-RNA       0.66      0.73      0.69        96
 B-cell_line       0.55      0.63      0.58       393
 B-cell_type       0.78      0.72      0.75      1730
   B-protein       0.74      0.80      0.77      4507
       I-DNA       0.83      0.78      0.80      1397
       I-RNA       0.74      0.78      0.76       156
 I-cell_line       0.65      0.70      0.67       792
 I-cell_type       0.86      0.78      0.81      2691
   I-protein       0.82      0.72      0.77      4222
           O       0.96      0.97      0.97     70962

    accuracy                           0.93     87803
   macro avg       0.76      0.75      0.75     87803
weighted avg       0.93      0.93      0.93     87803
```

### NCBI (Disease NER)
- **Entities**: 1 type (Disease)
- **Tags**: 2 BIO tags + O + PAD = 4 classes
- **Vocabulary**: 5,747 words
- **Characters**: 86 characters

**Performance Metrics:**
```
              precision    recall  f1-score   support

   B-Disease       0.79      0.87      0.83       720
   I-Disease       0.86      0.82      0.84       822
           O       0.99      0.99      0.99     19475

    accuracy                           0.98     21017
   macro avg       0.88      0.89      0.89     21017
weighted avg       0.98      0.98      0.98     21017
```

## 🎨 Color Legend for Tags

### JNLPBA Entities
| Entity | Tag | Color | HEX Code |
|--------|-----|-------|----------|
| **DNA** | B-DNA | Red | `#FF6B6B` |
| | I-DNA | Light Red | `#FF8E8E` |
| **RNA** | B-RNA | Cyan | `#4ECDC4` |
| | I-RNA | Light Cyan | `#7FDFD9` |
| **Protein** | B-protein | Blue | `#45B7D1` |
| | I-protein | Light Blue | `#7ACFE5` |
| **Cell Type** | B-cell_type | Green | `#96CEB4` |
| | I-cell_type | Light Green | `#B8E0CD` |
| **Cell Line** | B-cell_line | Brown | `#6D664F` |
| | I-cell_line | Light Brown | `#C39A12` |
| **Other** | O | Transparent | - |

### NCBI Entities
| Entity | Tag | Color | HEX Code |
|--------|-----|-------|----------|
| **Disease** | B-Disease | Red | `#FF6B6B` |
| | I-Disease | Light Red | `#FF8E8E` |
| **Other** | O | Transparent | - |


### Testing

```bash
# Run the application in development mode
streamlit run streamlit_app.py --server.headless true

# Check imports
python -c "from models.models import CombinatorialNER; print('Import OK')"
```

## 📝 Code Examples

### Programmatic Usage

```python
from streamlit_app import StreamlitNERPredictor
from streamlit_utils import load_all_components

# Load components
components = load_all_components(
    model_path="./checkpoints/JNLPBA/WE/best_model.pt",
    vocab_dir="./vocab/jnlpba",
    word2vec_path="./word2Vecembeddings/jnlpba_word2vec"
)

# Initialize predictor
predictor = StreamlitNERPredictor(
    components, 
    dataset_name='JNLPBA',
    use_char_cnn=False,
    use_char_lstm=False,
    use_attention=False,
    use_fc_fusion=False
)

# Make prediction
text = "The BRCA1 gene is associated with breast cancer."
predictions = predictor.predict(text)
entities = predictor.extract_entities(predictions)
```

## 🚨 Troubleshooting

### Common Issues

1. **"Model not found"**
   - Check paths in `load_jnlpba_components()` and `load_ncbi_components()`
   - Ensure `.pt` files exist

2. **Import errors**
   ```bash
   # Add parent directory to PYTHONPATH
   export PYTHONPATH="$PYTHONPATH:$(pwd)/.."
   ```

3. **Memory issues**
   - Reduce `max_seq_len` in `preprocess_tokens()`
   - Use `torch.no_grad()` for inferences

4. **Tags not displaying**
   - Enable debug mode in sidebar
   - Verify `ENTITY_COLORS_*` contains all tags

### Debug Logs

Enable debug mode in sidebar to see:
- Raw predictions
- Individual tags
- Extracted entities
- Technical information

## 📈 Performance Summary

### JNLPBA Results
- **Overall Accuracy**: 93%
- **Macro F1-Score**: 75%
- **Best performing**: I-cell_type (F1: 0.81)
- **Challenging**: B-cell_line (F1: 0.58)

### NCBI Results
- **Overall Accuracy**: 98%
- **Macro F1-Score**: 89%
- **B-Disease F1**: 0.83
- **I-Disease F1**: 0.84

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📚 References

1. **Original Paper Replicated:**
   - Yuan, C., Wang, J., Yang, W., & Zhang, Y. (2020). Combinatorial embedding based multi-channel adaptive attentive neural network for biomedical named entity recognition. Journal of Biomedical Informatics, 104. [DOI: 10.1016/j.jbi.2020.103392](https://doi.org/10.1016/j.jbi.2020.103392)

2. **Datasets:**
   - Kim, J. D., Ohta, T., Tsuruoka, Y., Tateisi, Y., & Collier, N. (2004). Introduction to the bio-entity recognition task at JNLPBA. Proceedings of the International Joint Workshop on Natural Language Processing in Biomedicine and its Applications.
   - Dogan, R. I., Leaman, R., & Lu, Z. (2014). NCBI disease corpus: a resource for disease name recognition and concept normalization. Journal of biomedical informatics, 47, 1-10.

3. **Frameworks:**
   - [PyTorch Documentation](https://pytorch.org/docs/)
   - [Streamlit Documentation](https://docs.streamlit.io/)

## 👥 Authors

- **OURAHMA Maroua**.
- **ZERHOUANI Oumaima**
- **ANEJJAR Wiame**

## 🙏 Acknowledgments

- JNLPBA and NCBI dataset maintainers
- PyTorch community
- Streamlit contributors
- Authors of the original research paper for their innovative approach

