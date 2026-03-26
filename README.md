# 🧠 Deep Learning for Tabular Column Understanding

A research-oriented deep learning project for **semantic classification of tabular data columns** (Excel/CSV). The model learns to predict column types (email, phone, price, etc.) from heterogeneous multi-modal inputs: text (headers + cell values), statistical features, and pattern-based features.

---

## 📋 Table of Contents

- [Problem Definition](#problem-definition)
- [Methodology](#methodology)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Experiments](#experiments)
- [Ablation Study](#ablation-study)
- [Interpretability](#interpretability)
- [Limitations & Future Work](#limitations--future-work)

---

## Problem Definition

**Task:** Given a single column from a tabular dataset, classify it into one of 8 semantic types:

| Type | Description | Example Values |
|------|-------------|---------------|
| `email` | Email addresses | `john@gmail.com`, `anna@yahoo.com` |
| `phone` | Phone numbers | `+1-555-123-4567`, `(212) 555-0199` |
| `price` | Monetary amounts | `$49.99`, `129.00€` |
| `id` | Identifiers/codes | `ID-00042`, `REF-ABC-123` |
| `date` | Dates/timestamps | `2024-03-15`, `15/03/2024` |
| `name` | Person names | `John Smith`, `Dupont, Marie` |
| `address` | Physical addresses | `123 Main St, New York` |
| `categorical` | Categorical values | `active`, `male`, `senior` |

**Challenge:** This is a **multi-modal classification** problem requiring the model to combine:
- **Text signals** from column headers and sample values
- **Statistical signals** (uniqueness, entropy, null ratio)
- **Pattern signals** (regex-based type detection ratios)

---

## Methodology

### Data Generation

We generate synthetic tabular columns using realistic templates for each column type. Each sample consists of:

```python
{
    "header": "client_email",           # Column header text
    "values": ["john@gmail.com", ...],  # Sample cell values
    "stats": {                          # Statistical features
        "n_unique": 120,
        "entropy": 3.4,
        "null_ratio": 0.02,
        "mean_length": 18.5
    },
    "patterns": {                       # Pattern-based features
        "is_email": 0.95,
        "is_phone": 0.0,
        "is_numeric": 0.0,
        "is_date": 0.0,
        "has_at": 0.95,
        "has_dot": 0.98
    }
}
```

### Data Pipeline

- **Train/Val/Test split:** 70% / 15% / 15%
- **Statistical feature normalization:** Z-score normalization computed on training set only (prevents data leakage)
- **Text tokenization:** WordPiece tokenization via DistilBERT tokenizer
- **Text format:** `"header: {header} [SEP] values: {v1}, {v2}, ..."`

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Column Input                        │
│  header + values (text)  │  stats  │  patterns       │
└──────────┬───────────────┴────┬────┴──────┬──────────┘
           │                    │           │
    ┌──────▼───────┐     ┌─────▼───────────▼─────┐
    │  DistilBERT  │     │   Feature Encoder      │
    │  [CLS] → 768 │     │   MLP: 10 → 64 → 64   │
    └──────┬───────┘     └──────────┬─────────────┘
           │                        │
    ┌──────▼────────────────────────▼──────┐
    │         Fusion Module                 │
    │   Concat + Project: 832 → 256        │
    │   (or Attention-based fusion)         │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────▼───────────────────────┐
    │        Classifier Head               │
    │   MLP: 256 → 128 → 8 (softmax)      │
    │   + Dropout regularization           │
    └──────────────────────────────────────┘
```

### Component Details

| Component | Implementation | Justification |
|-----------|---------------|---------------|
| **Text Encoder** | DistilBERT (pretrained) | Captures semantic meaning from headers/values; DistilBERT is 40% smaller than BERT with 97% performance |
| **Feature Encoder** | 2-layer MLP with ReLU + Dropout | Projects heterogeneous tabular features into a learned representation space |
| **Fusion** | Concatenation + Linear projection | Simple and effective; attention-based fusion available as alternative |
| **Classifier** | 2-layer MLP with softmax | Standard classification head with dropout for regularization |

### Training Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Optimizer | AdamW | Better weight decay handling than Adam |
| Learning rate | 2×10⁻⁵ | Standard for transformer fine-tuning |
| Weight decay | 0.01 | L2 regularization |
| Batch size | 32 | Balance between gradient quality and memory |
| Max epochs | 20 | With early stopping (patience=5) |
| Loss | Cross-entropy | Standard for multi-class classification |
| Grad clipping | max_norm=1.0 | Prevents gradient explosion |
| LR scheduler | ReduceLROnPlateau | Adaptive LR reduction on validation loss plateau |

---

## Project Structure

```
├── config/
│   └── config.yaml              # All hyperparameters (single source of truth)
├── src/
│   ├── data/
│   │   ├── generator.py         # Synthetic data generation
│   │   └── dataset.py           # Custom PyTorch Dataset + DataLoader
│   ├── model/
│   │   ├── text_encoder.py      # DistilBERT-based text encoding
│   │   ├── feature_encoder.py   # MLP for statistical/pattern features
│   │   ├── fusion.py            # Concat and Attention fusion modules
│   │   └── classifier.py        # Full multi-modal classifier
│   ├── training/
│   │   ├── trainer.py           # Manual training loop
│   │   └── early_stopping.py    # Early stopping mechanism
│   ├── evaluation/
│   │   ├── metrics.py           # Accuracy, F1, ROC-AUC computation
│   │   └── analysis.py          # Learning curves, confusion matrix, error analysis
│   └── interpretability/
│       └── feature_importance.py # Permutation-based feature importance
├── scripts/
│   ├── train.py                 # Main training pipeline
│   ├── evaluate.py              # Standalone evaluation
│   └── ablation.py              # Ablation study (5 configurations)
├── tests/
│   ├── test_data.py             # Data generation and dataset tests
│   ├── test_model.py            # Model component tests
│   └── test_training.py         # Training pipeline tests
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python -m scripts.train --config config/config.yaml
```

This will:
1. Generate synthetic dataset (4,000 samples, 500 per class)
2. Train the multi-modal classifier with early stopping
3. Evaluate on test set (accuracy, F1, ROC-AUC)
4. Generate visualizations (learning curves, confusion matrix)
5. Perform error analysis and feature importance analysis
6. Save model to `outputs/model.pt`

### Evaluation

```bash
python -m scripts.evaluate --model outputs/model.pt --config config/config.yaml
```

### Ablation Study

```bash
python -m scripts.ablation --config config/config.yaml
```

### Running Tests

```bash
python -m pytest tests/ -v
```

---

## Experiments

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification accuracy |
| **F1 (macro)** | Macro-averaged F1-score across all classes |
| **ROC-AUC** | One-vs-rest ROC-AUC, macro-averaged |

### Analysis Outputs

1. **Learning Curves:** Training vs validation loss/accuracy per epoch — diagnoses overfitting
2. **Confusion Matrix:** Per-class prediction heatmap — identifies systematic errors
3. **Error Analysis:** Qualitative inspection of misclassified examples
4. **Feature Importance:** Permutation-based importance of each feature group

---

## Ablation Study

We systematically remove each input modality to measure its contribution:

| Configuration | Description |
|--------------|-------------|
| `full_model` | All modalities enabled (baseline) |
| `no_header` | Header text removed |
| `no_values` | Cell values removed |
| `no_stats` | Statistical features removed |
| `no_patterns` | Pattern features removed |

### Expected Insights

- **Header removal** should cause the largest drop — headers carry strong semantic signals
- **Pattern removal** should impact types with clear patterns (email, phone, date)
- **Stats removal** may have modest impact — patterns capture similar information
- **Value removal** tests whether the model can classify from metadata alone

---

## Interpretability

### Permutation Feature Importance

We measure feature importance by permuting each feature group and measuring the accuracy drop:

- **Higher drop = more important feature group**
- Applied to statistical features and pattern features separately
- Provides global feature importance across the test set

### Attention-Based Fusion (Optional)

When using attention fusion, the model learns adaptive weights for text vs tabular features:
- Attention weights can be extracted to see which modality the model relies on per sample
- Useful for understanding edge cases and failure modes

---

## Limitations & Future Work

### Current Limitations

1. **Synthetic data only:** Real-world tabular columns have more noise, mixed types, and domain-specific patterns
2. **Fixed column types:** The 8-class taxonomy may not cover all real-world column types
3. **English-centric:** Text patterns and names are primarily English
4. **Single-column classification:** Does not consider inter-column relationships

### Future Directions

1. **Real-world evaluation:** Test on public CSV/Excel datasets (Kaggle, data.gov)
2. **Multi-language support:** Extend generators and patterns for multilingual data
3. **Hierarchical classification:** Coarse-to-fine type taxonomy (e.g., text → name → first_name)
4. **Cross-column context:** Use attention across columns within the same table
5. **Active learning:** Identify uncertain samples for human annotation
6. **Model distillation:** Distill the transformer-based model into a faster inference model

---

## Technical Choices & Justifications

| Choice | Justification |
|--------|---------------|
| PyTorch (manual training loop) | Full control over training dynamics; no black-box frameworks |
| DistilBERT over BERT | 40% smaller, 60% faster, retains 97% of BERT's performance |
| Synthetic data | Controlled evaluation; eliminates labeling noise; scalable |
| Z-score normalization | Standard approach; prevents feature scale domination |
| AdamW over Adam | Decoupled weight decay; better generalization |
| Cross-entropy loss | Standard for multi-class classification |
| Permutation importance | Model-agnostic; interpretable; no additional training needed |

---

## License

MIT License — see [LICENSE](LICENSE) for details.