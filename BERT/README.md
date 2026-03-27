# BERT News Classifier

Fine-tuned BERT model for news classification on the AG News dataset.

## Results

- **Accuracy**: 89.60%
- **Training Loss**: 0.34

## Quick Start

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./my_bert_news_model")
tokenizer = AutoTokenizer.from_pretrained("./my_bert_news_model")

inputs = tokenizer("Your news article here", return_tensors="pt", truncation=True, padding="max_length")
prediction = model(**inputs).logits.argmax(dim=-1)
```

## Training

Run `bert.ipynb` to train the model yourself.

## Requirements

```bash
pip install transformers datasets evaluate torch numpy
```

## Files

- `bert.ipynb` - Training notebook
- `my_bert_news_model/` - Saved model and tokenizer
- `bert_news_classifier/` - Training checkpoints
